from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from exect.helper import (
    get_unique_sorted_indices,
    create_label_mask
)
from torch.utils.data import Dataset


class ExectDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.DataFrame,
        propensities: np.ndarray,
        dataset,
        lf_filter: pd.DataFrame = None,
        label_frequencies: np.ndarray = None,
        label_mask: torch.Tensor = None,
    ):
        # dataset
        self.data = data.reset_index()
        self.label_names = labels.iloc[:, 0].values
        self.dataset = dataset
        self.propensities = propensities
        self.frequencies = label_frequencies
        self.label_mask = label_mask
        # reciprocal filter
        self.lf_filter = lf_filter
        """ self.lf_filter_dict = {}
        if self.lf_filter is not None:
            for idx, _ in tqdm(enumerate(data.values), total=len(data)):
                reciprocal_ids = torch.tensor([-999], dtype=torch.long)
                found_reciprocal_ids = self.lf_filter[
                    self.lf_filter.content_ind == self.data.iloc[idx]['index']
                ].label_ind.values.tolist()
                if found_reciprocal_ids:
                    found_reciprocal_ids = found_reciprocal_ids
                    reciprocal_ids = torch.tensor(
                        found_reciprocal_ids, dtype=torch.long
                    )
                self.lf_filter_dict[idx] = reciprocal_ids
 """
        self.content = data[dataset.content_col].values
        self.targets = self.data[dataset.target_col].tolist()


        # self.data["label_names"] = self.data[dataset.target_col]
        self.title = self.data[self.dataset.title_col] if dataset.title_col else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, val=False):
        # get the content
        title = self.title[index] if self.dataset.title_col is not None else None
        text = self.content[index]
        labels_id = self.data.iloc[index]["target_ind"]
        labels = self.label_names[labels_id]
        propensities = torch.tensor(self.propensities[labels_id], dtype=torch.float)
        frequencies = torch.tensor(self.frequencies)
        labels_id = torch.tensor(labels_id, dtype=torch.long)
        reciprocal_ids = torch.tensor([-999], dtype=torch.long)


        if self.lf_filter is not None:
            content_id = self.data.iloc[index]["index"]
            reciprocal_ids = self.lf_filter[
                self.lf_filter["content_ind"] == content_id
            ]["label_ind"].values
            # remove reciprocal labels from labels_id array
            if reciprocal_ids.size > 0:
                for rid in reciprocal_ids:
                    labels_id = np.delete(labels_id, np.where(labels_id == rid))

            reciprocal_ids = torch.tensor(reciprocal_ids, dtype=torch.long)

        return (
            title,
            text,
            labels,
            labels_id,
            reciprocal_ids,
            propensities,
            frequencies,
            self.label_mask
        )


class ExectTrainCollate:
    def __init__(self, tokenizer, max_length, label_max_length, n_classes):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_classes = n_classes
        self.label_max_length = label_max_length

    def __call__(self, batch):
        texts = []
        b_label_names = []
        b_label_ids = []
        b_reciprocal_ids = []
        b_propensities = []
        # y
        y = torch.zeros((len(batch), self.n_classes))
        for i, x in enumerate(batch):
            title, text, label_names, label_ids, reciprocal_ids, propensities, frequencies, _ = x

            if title is not None:
                text = [title, text]
            texts.append(text)
            y[i, label_ids] = 1.0
            b_label_names.append(label_names)
            b_label_ids.append(label_ids)
            b_reciprocal_ids.append(reciprocal_ids)
            b_propensities.append(propensities)

        b_label_ids = np.concatenate(b_label_ids)
        b_label_names = np.concatenate(b_label_names)
        b_propensities = np.concatenate(b_propensities)

        unique_sorted_indices = get_unique_sorted_indices(b_label_ids)

        b_label_ids = b_label_ids[unique_sorted_indices]
        b_label_names = b_label_names[unique_sorted_indices]
        b_propensities = b_propensities[unique_sorted_indices]
        frequencies = calculate_pos_weights(frequencies)
        frequencies = frequencies[unique_sorted_indices]

        

        tokenized_texts = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )


        tokenized_labels = self.tokenizer(
            b_label_names.tolist(),
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.label_max_length,
            return_tensors="pt",
        )

        non_zero_indices = torch.where(y.sum(0) > 0)[0]
        # TODO: Check if I need this line
        # y = y[:, unique_sorted_indices]
        y = y[:, non_zero_indices]

        return (
            tokenized_texts,
            tokenized_labels,
            y,
            b_propensities,
            b_reciprocal_ids,
            frequencies
        )


class ExectValCollate:
    def __init__(self, tokenizer, max_length, n_classes):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_classes = n_classes

    def __call__(self, batch):
        texts = []
        b_reciprocal_ids = []

        y = np.zeros((len(batch), self.n_classes))
        for i, x in enumerate(batch):
            # title, text, label_names, label_ids, reciprocal_ids, propensities, _ = x
            title, text, _, label_ids, reciprocal_ids, _, frequencies, label_mask = x

            if title is not None:
                text = [title, text]
            texts.append(text)
            y[i, label_ids] = 1.0
            b_reciprocal_ids.append(reciprocal_ids)

        tokenized_texts = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return (tokenized_texts, torch.tensor(y), b_reciprocal_ids, frequencies, label_mask)

def calculate_pos_weights(label_frequencies, scaling_factor=1.0, smoothing=1.0):
    """
    Calculate positive weights for BCEWithLogitsLoss based on label frequencies.
    
    Args:
    label_frequencies (list or np.array): Frequencies of each label
    scaling_factor (float): Factor to scale the weights (default: 1.0)
    smoothing (float): Smoothing factor to avoid division by zero (default: 1.0)
    
    Returns:
    torch.Tensor: Tensor of positive weights for each label
    """
    # Convert to numpy array if it's a list
    freq = np.array(label_frequencies)
    
    # Add smoothing to avoid division by zero
    freq = freq + smoothing
    
    # Calculate the reciprocal of frequencies
    inv_freq = 1. / freq
    
    # Normalize the weights
    pos_weights = inv_freq / np.max(inv_freq)
    
    # Apply scaling factor
    pos_weights = pos_weights * scaling_factor
    
    return torch.FloatTensor(pos_weights)