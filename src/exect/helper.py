import os
import random
import re
import zipfile
import json
import gzip
import shutil
from pathlib import Path
from typing import Type, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import torch
import torch.nn.functional as F
from rich.progress import Progress
from torch.optim import Adam, AdamW, SGD
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from scipy.sparse import csr_matrix

from exect.metrics import compute_inv_propesity

def _get_optimizer_params(model, lr, weight_decay=0.0):
    # param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


def get_optimizer(
    model,
    optimizer_name: str,
    learning_rate: float,
    eps: float,
    betas: list,
    weight_decay: float,
    **kwargs,
):
    parameter_groups = _get_optimizer_params(model, learning_rate)

    if optimizer_name.lower() == "adam":
        optimizer = Adam(
            parameter_groups,
            lr=learning_rate,
            eps=eps,
            betas=betas,
            weight_decay=weight_decay,
        )

    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(
            parameter_groups,
            lr=learning_rate,
            eps=eps,
            betas=betas,
            weight_decay=weight_decay,
        )

    if optimizer_name.lower() == "sgd":
        optimizer = SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=betas[0],
            weight_decay=weight_decay,
        )

    return optimizer


def get_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    max_learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    pct_start: float,
    **kwargs,
):
    if scheduler_name.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    if scheduler_name.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    elif scheduler_name.lower() == "cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_learning_rate,
            total_steps=num_training_steps,
            pct_start=pct_start,
        )

    return scheduler


def download_dataset(
    dataset: object,
    path: Type[Union[Path, str]] = Path.home() / ".exect-data",
    force_download=False,
):
    """
    It downloads a file from a URL and saves it to a specified path

    Args:
      dataset (dict): Dictionary containing the name and the url of a dataset.
      path (str): The path to the directory where the dataset will be downloaded. Defaults to ./datasets
      force_download: If True, the dataset will be downloaded even if it already exists in the path.
    Defaults to False
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    url = dataset.url
    zip_file_name = dataset.file_name
    dataset_name = dataset.name
    path = Path(path)

    if (Path(path) / dataset_name).exists() and not force_download:
        print(f"Dataset already exists.")
    elif "drive.google.com" in url:
        import gdown

        gdown.download(url, os.path.join(path, zip_file_name), quiet=False)
    else:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = total_size_in_bytes // 1000

        with Progress(transient=True) as progress:
            download_task = progress.add_task(
                f":arrow_down: Downloading {dataset_name}...", total=total_size_in_bytes
            )

            with open(os.path.join(path, zip_file_name), "wb") as file:
                for data in response.iter_content(block_size):
                    progress.update(download_task, advance=block_size)
                    file.write(data)


def unzip_dataset(
    dataset: object, path: Type[Union[Path, str]] = Path.home() / ".exect-data"
):
    zip_file_name = dataset.file_name

    if isinstance(path, str):
        path = Path(path)

    if (path / zip_file_name).exists():
        with zipfile.ZipFile(os.path.join(path, zip_file_name), "r") as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue

                source = zip_ref.open(member)
                os.makedirs(path / dataset.name, exist_ok=True)
                target = open(os.path.join(path / dataset.name, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

        os.remove(path / zip_file_name)


def get_criterion(distance: str, margin: float):
    assert distance in [
        "cosine",
        "euclidean",
    ], "Only 'cosine' and 'euclidean' distances are supported"
    if distance == "cosine":
        criterion = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=margin,
        )
    elif distance == "euclidean":
        criterion = torch.nn.TripletMarginLoss(margin=margin)

    return criterion


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_embeddings(model, val_dl, config, label=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = []
    tokenizer = config["tokenizer"]
    if label:
        content = list(val_dl.dataset.labels.values)
        for i in range(0, len(content), config["batch_size"]):
            content_tokens = tokenizer(
                content[i : i + config["batch_size"]],
                add_special_tokens=True,
                padding="max_length",
                max_length=25,
                return_tensors="pt",
            )
            content_tokens = to_device(content_tokens, device)
            embeddings.append(model.forward_label(content_tokens).cpu())

    else:
        max_length = config["content_max_tokens"]
        content = val_dl.dataset.content.tolist()
        for i in range(0, len(content), config["batch_size"]):
            content_tokens = tokenizer(
                content[i : i + config["batch_size"]],
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            content_tokens = to_device(content_tokens, device)
            embeddings.append(model.forward_content(content_tokens).cpu())

    embeddings = torch.cat(embeddings, dim=0)

    return embeddings


def to_device(dictionary, device=None):
    """
    It takes a dictionary of tensors and moves them to the device specified by the `device` argument

    Args:
      dictionary: a dictionary of tensors
      device: The device to run the model on.

    Returns:
      A dictionary with the keys 'x', 'y', 'x_lengths', and 'y_lengths'
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for key in dictionary:
        dictionary[key] = dictionary[key].to(device)
    return dictionary


def calculate_distance(content_embeddings, label_embeddings, distance):
    assert distance in [
        "cosine",
        "inner_product",
    ], "--distance must be either 'euclidean' or 'cosine'"
    if distance == "inner_product":
        return content_embeddings @ label_embeddings.T
    elif distance == "cosine":
        return cosine_matrix(content_embeddings, label_embeddings)


def tokenize_labels(labels, config):
    tokenizer = config["tokenizer"]
    return tokenizer(
        labels,
        add_special_tokens=True,
        padding="max_length",
        max_length=config["label_max_tokens"],
        truncation=True,
        return_tensors="pt",
    )


def tokenize_text(text, config):
    if isinstance(text, list):
        if text[0] == None:
            text = text[1]
            return _tokenize_string(text, config)
        else:
            return _tokenize_list(text, config)
    else:
        return _tokenize_string(text, config)


def _tokenize_list(text, config):
    tokenizer = config["tokenizer"]
    return tokenizer(
        *text,
        add_special_tokens=True,
        max_length=config["content_max_tokens"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def _tokenize_string(text, config):
    tokenizer = config["tokenizer"]
    return tokenizer(
        text,
        add_special_tokens=True,
        max_length=config["content_max_tokens"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def _get_paramater_groups(model, lr, lr_mult=0.99):
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
    layer_names.reverse()

    parameters = []
    for idx, name in enumerate(layer_names):
        # append layer parameters
        parameters += [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n == name and p.requires_grad
                ],
                "lr": lr,
            }
        ]

        # update learning rate
        lr *= lr_mult

    return parameters


def replace_digits(text):
    return re.sub("\d+", "[DIG]", text)


def strip_multiple_whitespace(text):
    return re.sub("\s+", " ", text)


def convert_similarities_to_preds(similarities, labels, k):
    """
    Takes the similarities between label and content embedding and converts it into a one-hot encoding,
    where the values at the corresponding indices of the top k most similar labels are set to 1.

    Args:
      similarities: the similarities between content and label embeddings
      labels: the ground truth labels
      k: the number of nearest neighbors to use for prediction

    Returns:
        Top k similarities one hot encoded.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = torch.topk(similarities, k=k)[1]
    preds = torch.zeros(labels.shape).to(device).to(dtype=torch.float16)
    preds = torch.where(preds.scatter_(1, index, similarities) != 0.0, 1.0, 0.0)
    return preds


def cosine_matrix(a, b, eps=1e-8):
    """
    > It takes two matrices, normalizes them, and returns the cosine similarity matrix

    Args:
      a: the first matrix
      b: batch size
      eps: a small value to avoid division by zero

    Returns:
      The cosine similarity matrix between the two sets of vectors.
    """

    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def read_jsonl(filepath, usecols=None) -> pd.DataFrame:
    """
    Convert a gziped JSON lines file to pandas object.

    Args:
    filepath : path to a gizped JSON file, for example: ``/path/to/table.json.gz``.
    usecols : list of str, optional. Specifies which columns to read from the JSON file. If not specified, all columns will be read.
    """
    with gzip.open(filepath, "r") as f:
        data = []

        for line in tqdm(f):
            doc = json.loads(line)
            if usecols == None:
                usecols = doc.keys()
            line_data = [doc[c] for c in usecols]
            data.append(line_data)

        df = pd.DataFrame(data, columns=usecols)

    return df


def calculate_propensity_scores(data: pd.DataFrame, dataset_name: str, num_classes: int):
    # Propensities
    # Jain, H., Prabhu, Y., & Varma, M. (2016, August). Extreme multi-label loss functions for recommendation, tagging, ranking & other missing label applications. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 935-944).
    if "wiki" in dataset_name.lower():
        A = 0.5
        B = 0.4
    elif "amazon" in dataset_name.lower():
        A = 0.6
        B = 2.6
    else:
        A = 0.55
        B = 1.5

    rows, cols = [], []
    for i, row in enumerate(data["target_ind"].values.tolist()):
        rows.append([i] * len(row))
        cols.append(row)
    rows = np.array([item for sublist in rows for item in sublist])
    cols = np.array([item for sublist in cols for item in sublist])
    vals = np.ones_like(rows)

    label_matrix = csr_matrix((vals, (rows, cols)), shape=(data.shape[0], num_classes))

    return compute_inv_propesity(label_matrix, A, B)


def sort_accordingly(a: np.array, b: np.array) -> tuple:
    """
    Sorts two arrays 'a' and 'b' according to the values in array 'a'.

    Args:
        a (np.array): The first array.
        b (np.array): The second array.

    Returns:
        tuple: A tuple containing the sorted arrays 'a' and 'b'.
    """
    assert len(a) == len(b)

    sorted_indices = np.argsort(a)
    return a[sorted_indices], b[sorted_indices]


def remove_duplicates_accordingly(a: np.array, b: np.array) -> tuple:
    """Remove duplicates from array 'a' and remove corresponding values from array 'b'.

    Args:
        a (np.array): The input array 'a'.
        b (np.array): The input array 'b'.

    Returns:
        tuple: A tuple containing the modified arrays 'a' and 'b'.

    Raises:
        AssertionError: If the lengths of arrays 'a' and 'b' are not equal.
    """
    assert len(a) == len(b)

    a, unique_indices = np.unique(a, return_index=True)
    return a, b[unique_indices]


def remove_zero_sum_columns(arr):
    column_sums = np.sum(arr, axis=0)
    nonzero_columns = column_sums != 0
    new_array = arr[:, nonzero_columns]
    return new_array


def get_unique_sorted_indices(arr):
    sorted_indices = np.argsort(arr)
    unique = np.unique(arr[sorted_indices], return_index=True)[1]
    unique_sorted_indices = sorted_indices[unique]
    return unique_sorted_indices


def create_label_embeddings(trainer, model):
        # check if trainer has attr
        if trainer.train_dataloader:
            print('Has train_dataloader')
            label_names = trainer.train_dataloader.dataset.label_names.tolist()
            batch_size = trainer.train_dataloader.batch_size

        else:
            print('Has test_dataloader')
            label_names = trainer.test_dataloaders.dataset.label_names.tolist()
            batch_size = trainer.test_dataloaders.batch_size

        label_max_length = trainer.datamodule.label_max_length
        tokenized_labels = model.tokenizer(
            label_names,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=label_max_length,  # TODO: Remove hardcoded value
            return_tensors="pt",
        )
        print(f"Batch size: {batch_size}")
        model.eval()
        with torch.no_grad():
            label_embeddings = []
            # loop over label names and create embeddings
            for i in tqdm(
                range(0, len(label_names) - 1, batch_size), desc="Encoding Labels..."
            ):
                input_ids = tokenized_labels["input_ids"][i : i + batch_size]
                attention_mask = tokenized_labels["attention_mask"][i : i + batch_size]
                inputs = to_device(
                    {"input_ids": input_ids, "attention_mask": attention_mask}
                )
                label_embedding = model.encode_label(inputs)
                label_embeddings.append(label_embedding)
            label_embeddings = torch.cat(label_embeddings, dim=0)
        print("Encoding successful...")
        return label_embeddings



def replace_similarities(A, B):
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    unique, counts = np.unique(A, return_counts=True)
    duplicates = unique[counts > 1]

    for value in duplicates:
        indices = np.where(A == value)[0]
        max_index = indices[np.argmax(np.array(B)[indices])]
        for i in indices:
            if i != max_index:
                A[i] = 0

    return torch.tensor(A, dtype=torch.half)


def scale_tensor(tensor, min_val=-6, max_val=6, dim=1):
    if dim is not None:
        # Compute min and max along the specified dimension
        current_min = tensor.min(dim=dim, keepdim=True)[0]
        current_max = tensor.max(dim=dim, keepdim=True)[0]
    else:
        current_min = tensor.min()
        current_max = tensor.max()

    # Normalize tensor to [0, 1]
    normalized_tensor = (tensor - current_min) / (current_max - current_min)

    # Scale to [min_val, max_val]
    scaled_tensor = (normalized_tensor * (max_val - min_val)) + min_val

    return scaled_tensor


def create_label_mask(label_embeddings, label_frequencies):
    n_labels, dim_emb = label_embeddings.shape
    
    # Create a mask tensor initialized with ones
    mask = torch.ones(n_labels, dtype=torch.float32)
    
    # Create a dictionary to store the frequency of each unique label embedding
    label_freq_dict = {}
    
    # Iterate over the label embeddings and frequencies
    for i in range(n_labels):
        label_emb = label_embeddings[i]
        freq = label_frequencies[i]
        
        # Convert the label embedding to a tuple for hashing
        label_emb_tuple = tuple(label_emb.tolist())
        
        # Check if the label embedding exists in the dictionary
        if label_emb_tuple in label_freq_dict:
            # If the current frequency is lower than the stored frequency,
            # set the corresponding mask value to zero
            if freq < label_freq_dict[label_emb_tuple]:
                mask[i] = 0.0
        else:
            # If the label embedding is not in the dictionary, add it with its frequency
            label_freq_dict[label_emb_tuple] = freq
    
    return mask