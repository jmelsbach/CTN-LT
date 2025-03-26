from pathlib import Path

import pandas as pd
from exect.datasets import load_dataset
from exect.helper import (
    calculate_propensity_scores,
    download_dataset,
    read_jsonl,
    unzip_dataset,
    create_label_mask
)
from transformers import AutoTokenizer
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from exect.exect_data import ExectDataset, ExectTrainCollate, ExectValCollate
import numpy as np
import csv
from lightning.pytorch.utilities import rank_zero_only
import os

class ExectDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str = "Wiki10-31K",
        val_size: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 128,
        label_max_length: int = 16,
        model: str = "distilbert-base-uncased",
        replace_label_underscore: bool = False,
        fold: int = -1,
    ):
        super().__init__()

        self.dataset_dir = Path.home() / ".exect-data"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.dataset = load_dataset(dataset)
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.label_max_length = label_max_length
        self.model = model
        self.replace_label_underscore = replace_label_underscore
        self.sampler = None
        self.fold = fold

    def prepare_data(self):
        download_dataset(self.dataset, path=self.dataset_dir)
        unzip_dataset(self.dataset, path=self.dataset_dir)

    def setup(self, stage=None, force_reload=False):
        train_path = self.dataset_dir / self.dataset.name / self.dataset.train_file
        test_path = self.dataset_dir / self.dataset.name / self.dataset.test_file

        usecols = [
            self.dataset.content_col,
            self.dataset.target_col,
            self.dataset.title_col,
        ]

        usecols = [col for col in usecols if col is not None]

        train_df = read_jsonl(train_path, usecols=usecols)
        test_df = read_jsonl(test_path, usecols=usecols)

        if self.dataset.label_file:
            csv_keys = [".txt", ".csv"]
            json_keys = [".json", "jsonl"]
            if any(key in self.dataset.label_file for key in csv_keys):
                self.label_df = pd.read_csv(
                    self.dataset_dir / self.dataset.name / self.dataset.label_file,
                    header=None,
                    encoding=self.dataset.encoding,
                    sep="\t",
                    quoting=csv.QUOTE_NONE,  # Disable special handling of quotes
                    escapechar=None,  # No escape character
                    quotechar=None,  # No quote character
                    lineterminator='\n'  # Explicitly set line terminator,
                )
            elif any(key in self.dataset.label_file for key in json_keys):
                self.label_df = pd.read_json(
                    self.dataset_dir / self.dataset.name / self.dataset.label_file,
                    lines=self.dataset.jsonl,
                    compression="infer",
                )
        # If there is a specific label selected, use this for textual representation
        # of label
        if self.dataset.label_content_col:
            self.label_df = pd.DataFrame(self.label_df[self.dataset.label_content_col])
            if self.replace_label_underscore:
                self.label_df = self.label_df.apply(lambda x: x.str.replace("_", " "))

        self.train_lf_filter = None
        if self.dataset.train_lf_filter:
            train_lf_filter_df = pd.read_csv(
                self.dataset_dir / self.dataset.name / self.dataset.train_lf_filter,
                header=None,
                encoding="latin-1",
                sep=" ",
            )
            train_lf_filter_df.columns = ["content_ind", "label_ind"]
            self.train_lf_filter = train_lf_filter_df

        self.test_lf_filter = None
        if self.dataset.test_lf_filter:
            test_lf_filter = pd.read_csv(
                self.dataset_dir / self.dataset.name / self.dataset.test_lf_filter,
                header=None,
                encoding="latin-1",
                sep=" ",
            )
            test_lf_filter.columns = ["content_ind", "label_ind"]
            self.test_lf_filter = test_lf_filter

        self.frequencies = np.zeros(len(self.label_df))
        for label_set in train_df[self.dataset.target_col].values:
            self.frequencies[label_set] += 1

        self.save_frequencies()

        tokenized_labels = self.tokenizer(
            self.label_df.iloc[:,0].values.tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.label_max_length,
        )

        self.label_mask = create_label_mask(tokenized_labels["input_ids"], self.frequencies)

        n_duplicates = len(self.label_df) - int(self.label_mask.sum())
        if n_duplicates > 0:
            print(f"Found {n_duplicates} duplicate labels")

        self.propensities = calculate_propensity_scores(
            train_df, self.dataset.name, num_classes=self.dataset.num_classes
        )

        # train_df = train_df.sample(frac=self.dataset_pct).reset_index(drop=True)

        # create 5 fold

        if self.fold in range(5):
            train_df["fold"] = -1
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True)
            for i, (train_index, test_index) in enumerate(kf.split(train_df)):
                # Correctly assign 'fold' values using .loc
                train_df.loc[train_df.index[test_index], "fold"] = i
            val_df = train_df[train_df["fold"] == self.fold]
            train_df = train_df[train_df["fold"] != self.fold]

        else:
            if self.val_size > 0:
                train_df, val_df = train_test_split(train_df, test_size=self.val_size)
            else:
                # Create empty validation set from training set
                val_df = train_df.iloc[:0, :].copy()

        self.train_dataset = ExectDataset(
            data=train_df,
            labels=self.label_df,
            propensities=self.propensities,
            dataset=self.dataset,
            lf_filter=None,
            label_frequencies=self.frequencies,
            label_mask=self.label_mask,
        )

        self.val_dataset = ExectDataset(
            data=val_df,
            labels=self.label_df,
            propensities=self.propensities,
            dataset=self.dataset,
            lf_filter=self.train_lf_filter,
            label_frequencies=self.frequencies,
            label_mask=self.label_mask,
        )

        self.test_dataset = ExectDataset(
            data=test_df,
            labels=self.label_df,
            propensities=self.propensities,
            dataset=self.dataset,
            lf_filter=self.test_lf_filter,
            label_frequencies=self.frequencies,
            label_mask=self.label_mask,
        )

        self.train_collate_fn = ExectTrainCollate(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label_max_length=self.label_max_length,
            n_classes=self.dataset.num_classes,
        )

        self.val_collate_fn = ExectValCollate(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            n_classes=self.dataset.num_classes,
        )

    def train_dataloader(self):
        if self.sampler:
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=self.train_collate_fn,
                num_workers=self.num_workers,
                sampler=self.sampler,
            )

        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=self.train_collate_fn,
                num_workers=self.num_workers,
                shuffle=True,
            )
        return

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_collate_fn,
            num_workers=self.num_workers,
        )

    @rank_zero_only
    def save_frequencies(self):
        model_dir = os.path.join("models", self.dataset.name)
        os.makedirs(model_dir, exist_ok=True)
        filename = "frequencies.npy"
        save_path = os.path.join(model_dir, filename)
        np.save(save_path, self.frequencies)
        print(f"Frequencies saved to {save_path}")
