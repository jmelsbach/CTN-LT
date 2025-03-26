from typing import Any, Optional
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from helper import to_device, create_label_embeddings
import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sampler import SortedClusterSampler
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz


class EmbeddingCallback(Callback):
    def on_validation_epoch_start(self, trainer, model):
        label_embeddings = create_label_embeddings(trainer, model)
        model.label_embeddings = label_embeddings

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, model: pl.LightningModule
    ) -> None:
        model.label_embeddings = None

    def on_test_epoch_start(self, trainer, module) -> None:
        self.on_validation_epoch_start(trainer, module)

class LabelClusterCallback(Callback):
    def on_train_epoch_end(self, trainer, model) -> None:
        label_embeddings = create_label_embeddings(trainer, model)
        target_ind: list[
            list
        ] = trainer.train_dataloader.dataset.data.target_ind.tolist()
        label_embeddings_avg = []
        for t in tqdm(
            target_ind, desc="Averaging label embeddings", total=len(target_ind)
        ):
            label_embeddings_avg.append(
                label_embeddings[t].mean(dim=0).cpu().detach().numpy()
            )
        label_embeddings = np.array(label_embeddings_avg)
        kmeans = KMeans(n_clusters=100, random_state=0, n_init="auto").fit(
            label_embeddings
        )
        kmeans_labels = kmeans.labels_
        trainer.datamodule.sampler = SortedClusterSampler(kmeans_labels)


class SavePredictionCallback(Callback):

    def __init__(self):
        self.predictions = {
            'text': [],
            'labels': [],
            'label_ids': [],
            'similarities': [],
            'y': []
        }

    # def on_validation_batch_end(
    #     self,
    #     trainer: Trainer,
    #     pl_module: LightningModule,
    #     outputs: STEP_OUTPUT | None,
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ) -> None:
    #     label_names = trainer.train_dataloader.dataset.label_names
    #     text, tokenized_text, y, reciprocal_ids, frequencies, label_mask = batch
    #     text, preds = outputs
    #     preds *= label_mask
    #     label_similarities, label_ids = preds.topk(100, dim=1)
    #     y = self.find_indices_of_ones_per_row(y)


    #     self.predictions['text'] += text
    #     self.predictions['labels'] += label_names[label_ids.cpu().detach().numpy().tolist()].tolist()
    #     self.predictions['label_ids'] += label_ids.cpu().detach().numpy().tolist()
    #     self.predictions['similarities'] += label_similarities.cpu().detach().numpy().tolist()

    #     self.predictions['y'] += ([label_names[i].tolist() for i in y])

    # def on_validation_epoch_end(
    #     self, trainer: pl.Trainer, pl_module: pl.LightningModule
    # ) -> None:
    #     df = pd.DataFrame.from_dict(self.predictions)
    #     df.to_csv('predictions.csv')
    #     print('saved predictions')

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        label_names = trainer.test_dataloaders.dataset.label_names
        text, tokenized_text, y, reciprocal_ids, _, label_mask = batch
        text, preds = outputs
        preds *= label_mask
        label_similarities, label_ids = preds.topk(100, dim=1)
        y = self.find_indices_of_ones_per_row(y)


        self.predictions['text'] += text
        self.predictions['labels'] += label_names[label_ids.cpu().detach().numpy().tolist()].tolist()
        self.predictions['label_ids'] += label_ids.cpu().detach().numpy().tolist()
        self.predictions['similarities'] += label_similarities.cpu().detach().numpy().tolist()

        self.predictions['y'] += ([label_names[i].tolist() for i in y])

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        df = pd.DataFrame.from_dict(self.predictions)
        df.to_csv('predictions.csv')
        print('saved predictions')

    def find_indices_of_ones_per_row(self, tensor):
        # List to store the indices of ones for each row
        indices_list = []

        # Iterate through each row
        for row in tensor:
            # Find the indices in this row where the value is 1
            indices = (row == 1).nonzero(as_tuple=False).squeeze().tolist()

            # Ensure the output is a list of lists
            if not isinstance(indices, list):
                indices = [indices]

            indices_list.append(indices)

        return indices_list



class SaveSparsePredictionCallback(Callback):

    def __init__(self,name: str = "checkpoint", topk: int = 100):
        self.name = name
        self.topk = topk
        self.similarities = []
        self.indices = []
        self.y = []
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        propensities = trainer.datamodule.propensities
        np.save(f"propensities/{self.name}_propensities.npy", propensities)
        print('propensities saved')
    
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, y, _, _, label_mask = batch
        preds = outputs
        preds *= label_mask
        label_similarities, label_ids = preds.topk(self.topk, dim=1)
        self.similarities.append(label_similarities.cpu().detach())
        self.indices.append(label_ids.cpu().detach())
        self.n_labels = preds.shape[1]
        self.y.append(y.cpu().detach())

    

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        propensities = trainer.datamodule.propensities
        np.save(f"{self.name}_propensities.npy", propensities)
        n_rows = len(torch.cat(self.similarities))
        similarities = torch.cat(self.similarities).numpy().flatten()
        cols = torch.cat(self.indices).numpy().flatten()
        y = csr_matrix(torch.cat(self.y).numpy())
        #convert y to float16
        rows = np.array(self._create_list(n_rows, self.topk))
        label_matrix = csr_matrix((similarities, (rows, cols)), shape=(n_rows, self.n_labels))
        # Save the label matrix as npz and numpy
        save_npz("preds.npz", label_matrix)
        np.save(f"preds.npy", label_matrix.toarray())
        # Save the y matrix as npz and numpy
        save_npz(f"targets.npz", y)
        np.save(f"targets.npy", y.toarray())
    
    def _create_list(self, n_rows, top_k):
        topk_list = []
        for i in range(n_rows):
            topk_list.append([i] * top_k)
        return np.array(topk_list).flatten()