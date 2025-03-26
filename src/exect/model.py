import numpy as np
import torch
import torch.nn as nn
from exect.helper import (
    calculate_distance,
    get_optimizer,
    get_scheduler,
    scale_tensor,
)
from lightning import pytorch as pl
from typing import Union, List
from exect.loss import MaskedBCEWithLogitsLoss
from exect.loss import MultiLabelCrossEntropyLoss
from exect.metrics import ndcg, precision, psndcg, psprecision, recall
from scipy import sparse
from transformers import AutoConfig, AutoModel, AutoTokenizer
from loss_scheduler import LossScheduler


class ExectModel(pl.LightningModule):
    def __init__(
        self,
        model: str = "distilbert-base-uncased",
        distance: str = "inner_product",
        propensity_loss: bool = True,
        scheduler: str = "cycle",
        learning_rate: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 0.00001,
        weight_decay: float = 0.01,
        optimizer: str = "adamw",
        pct_start: float = 0.3,
        num_cycles: float = 0.5,
        num_warmup_steps: int = 0,
        n_mask: Union[int, List[int]] = [0],
        n_mask_max: int = 0,
        n_mask_min: int = 0,
        head: bool = False,
        pooling: str = "none",
        architecture: str = "tower",
        unfreeze_after: int = 0,
        skip_dropout: float = 0.0,
        dropout_head: float = 0.0,
        loss_fn: str = "BCE",
        min_frequency: int = -1,
        lora: bool = False,
        alpha: float = 0.8,
        logit_scaling: bool = False
    ):
        """_summary_

        Args:
            model (str, optional): Pretrained model checkpoint. Defaults to "distilbert-base-uncased".
            distance (str, optional): Metric for distance calculation between to embeddings. Defaults to "cosine".
            propensity_loss (bool, optional): Whether or not to use propensity loss. Defaults to False.
            scheduler (str, optional): Name of the scheduler used for training. Defaults to "cycle".
            learning_rate (float, optional): Base learning rate for training. Defaults to 1e-5.
            betas (list[float], optional): Betas for Adam optimizer. Defaults to (0.9, 0.99).
            weight_decay (float, optional): Weight decay for Adam optimizer. Defaults to 0.01.
            optimizer (str, optional): Optimizer for training. Defaults to "adamw".
            pct_start (float, optional): Percentage of steps for the warmup of the scheduler. Defaults to 0.3.
            num_cycles (float, optional): Number of cycles for the onecycle scheduler. Defaults to 0.5.
            num_warmup_steps (int, optional): Number of warmup steps. Defaults to 0.
            eps (float, optional): Eps for Adam optimizer. Defaults to 1e-5.
            n_mask (int, optional): Number of top losses that don't get masked. Defaults to 0.
            head (bool, optional): Whether or not to use a head. Defaults to False.
            pooling (str, optional): Pooling strategy. Defaults to "none".
            architecture (str, optional): Architecture of the model. Defaults to "tower".
            unfreeze_after (int, optional): Epoch after which to unfreeze the encoders. Defaults to 0.
            skip_dropout (float, optional): Skip Dropout value of SkipConnectionHead
            dropout_head (float, optional): Dropout Value for SkipConnectionHead
            loss_fn (str, optional): Loss function. Defaults to "BCE".
            min_frequency (int, optional): Minimum frequency for a label to be considered. Defaults to -1.
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters()

        if isinstance(self.hparams.n_mask, int):
            self.hparams.n_mask = [self.hparams.n_mask]

        if self.hparams.n_mask_min > self.hparams.n_mask_max:
            self.hparams.n_mask_min, self.hparams.n_mask_max = (
                self.hparams.n_mask_max,
                self.hparams.n_mask_min,
            )
        self.model_config = AutoConfig.from_pretrained(model)
        # used in EmbeddingCallback and Inference

        if hasattr(self.model_config, "dim"):
            self.transformer_output_dim = self.model_config.dim
        elif hasattr(self.model_config, "hidden_size"):
            self.transformer_output_dim = self.model_config.hidden_size

        self.check_pooling()

        self.text_encoder = AutoModel.from_pretrained(
            model, output_hidden_states=self.output_hidden_states
        )
        self.label_encoder = AutoModel.from_pretrained(
            model, output_hidden_states=self.output_hidden_states
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.temperature = nn.Parameter(torch.randn(1), requires_grad=True)

        if self.hparams.lora:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                target_modules=["q_lin", "v_lin"], r=1, lora_alpha=32, lora_dropout=0.1
            )
            self.text_encoder = get_peft_model(self.text_encoder, peft_config)
            self.label_encoder = get_peft_model(self.label_encoder, peft_config)
            self.text_encoder.print_trainable_parameters()

        if self.hparams.head:
            self.text_head = SkipConnectionHead(
                self.model_config.hidden_size, dropout=self.hparams.dropout_head
            )
            if self.hparams.architecture == "tower":
                self.label_head = SkipConnectionHead(
                    self.model_config.hidden_size, dropout=self.hparams.dropout_head
                )
        else:
            self.text_head = nn.Identity()
            if self.hparams.architecture == "tower":
                self.label_head = nn.Identity()

        if self.hparams.pooling == "concat":
            self.output_hidden_states = True
            self.projection_layer = nn.Linear(
                self.transformer_output_dim * 5, self.transformer_output_dim
            )

        if self.hparams.unfreeze_after > 0:
            self._freeze_encoders()

        # metrics
        self._initialize_metrics()

    def check_pooling(self):
        if self.hparams.pooling == "concat":
            self.output_hidden_states = True
            self.projection_layer = nn.Linear(
                self.transformer_output_dim * 5, self.transformer_output_dim
            )

        else:
            self.output_hidden_states = False

    def encode_text(self, text):
        model_output = self.text_encoder(**text)

        embedding = self._pool(
            model_output=model_output,
            attention_mask=text["attention_mask"],
            strategy=self.hparams.pooling,
        )

        return self.text_head(embedding)

    def encode_label(self, labels):
        if self.hparams.architecture == "siamese":
            return self.encode_text(labels)

        model_output = self.label_encoder(**labels)

        embedding = self._pool(
            model_output=model_output,
            attention_mask=labels["attention_mask"],
            strategy=self.hparams.pooling,
        )

        return self.label_head(embedding)

    def on_train_epoch_start(self) -> None:
        if (
            self.current_epoch == self.hparams.unfreeze_after
            and self.hparams.unfreeze_after != 0
        ):
            self._unfreeze_encoders()
            print(f"Encoders unfrozen after {self.current_epoch} epochs.")

    def training_step(self, batch, batch_idx):
        if not hasattr(self, "topk_scheduler"):
            self.topk_scheduler = LossScheduler(
                start=self.hparams.n_mask_max,
                end=self.hparams.n_mask_min,
                max_steps=self.trainer.estimated_stepping_batches,
            )

        text, labels, y, propensities, _, frequencies = batch
        encoded_text = self.encode_text(text)
        encoded_labels = self.encode_label(labels)

        self.log("Number of labels in batch", encoded_labels.shape[0])

        y_hat = calculate_distance(encoded_text, encoded_labels, self.hparams.distance)

        ### JUST FOR LOGGING/DEBUGGING ###
        with torch.no_grad():
            self.log("y_hat_mean", y_hat.mean())
            self.log("y_hat_max", y_hat.max())
            self.log("y_hat_min", y_hat.min())
            self.log("y_hat_true_mean", y_hat[y == 1].mean())
            self.log("y_hat_true_max", y_hat[y == 1].max())
            self.log("y_hat_true_min", y_hat[y == 1].min())

        #y_hat = scale_tensor(y_hat, min_val=-7, max_val=7, dim=1)
        if self.hparams.distance == "cosine":
            scaler = torch.exp(self.temperature.to(self.device).requires_grad_(True))
            y_hat /= scaler
            self.log("scaler", scaler)

        self.current_n_mask = self.topk_scheduler.step(current_step=self.global_step)
        self.log("n_mask", self.current_n_mask)

        loss = self._calc_loss(y_hat, y, frequencies)
        self.log("Train Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokenized_text, y, reciprocal_ids, frequencies, label_mask = batch
        propensities = self.trainer.datamodule.propensities
        encoded_text = self.encode_text(tokenized_text)

        preds = calculate_distance(
            encoded_text, self.label_embeddings, self.hparams.distance
        )

        for i, ids in enumerate(reciprocal_ids):
            ids = ids.detach().cpu().numpy()
            ids = ids[ids >= 0]
            preds[i, ids] = torch.tensor([-999.0], dtype=preds.dtype)

        preds = preds * (frequencies >= self.hparams.min_frequency)

        preds = preds * label_mask

        self._calc_metrics(preds, y, propensities, test=False)

        return  preds

    def test_step(self, batch, batch_idx):
        tokenized_text, y, reciprocal_ids, frequencies, label_mask = batch
        propensities = self.trainer.datamodule.propensities

        encoded_text = self.encode_text(tokenized_text)
        preds = calculate_distance(
            encoded_text, self.label_embeddings, self.hparams.distance
        )

        for i, ids in enumerate(reciprocal_ids):
            ids = ids.detach().cpu().numpy()
            ids = ids[ids >= 0]
            preds[i, ids] = torch.tensor([-999.0], dtype=preds.dtype)

        # TODO replace hardcoded 30
        preds = preds * (frequencies > self.hparams.min_frequency)
        preds = preds * label_mask
        self._calc_metrics(preds, y, propensities, test=True)

        return preds

    def on_validation_epoch_end(self):
        self.log("Precision@1", np.array(self.p1).mean(), sync_dist=True)
        self.log("Precision@3", np.array(self.p3).mean(), sync_dist=True)
        self.log("Precision@5", np.array(self.p5).mean(), sync_dist=True)

        self.log("Recall@5", np.array(self.r5).mean(), sync_dist=True)
        self.log("Recall@10", np.array(self.r10).mean(), sync_dist=True)
        self.log("Recall@20", np.array(self.r20).mean(), sync_dist=True)

        self.log("NDCG@1", np.array(self.n1).mean(), sync_dist=True)
        self.log("NDCG@3", np.array(self.n3).mean(), sync_dist=True)
        self.log("NDCG@5", np.array(self.n5).mean(), sync_dist=True)

        self.log("PSPrecision@1", np.array(self.psp1).mean(), sync_dist=True)
        self.log("PSPrecision@3", np.array(self.psp3).mean(), sync_dist=True)
        self.log("PSPrecision@5", np.array(self.psp5).mean(), sync_dist=True)

        self.log("PSNDCG@1", np.array(self.psn1).mean(), sync_dist=True)
        self.log("PSNDCG@3", np.array(self.psn3).mean(), sync_dist=True)
        self.log("PSNDCG@5", np.array(self.psn5).mean(), sync_dist=True)

        print(f"Validation Precision@1: {np.mean(self.p1)}")
        print(f"Validation Precision@3: {np.mean(self.p3)}")
        print(f"Validation Precision@5: {np.mean(self.p5)}")
        print(f"Validation Recall@5: {np.mean(self.r5)}")
        print(f"Validation Recall@10: {np.mean(self.r10)}")
        print(f"Validation Recall@20: {np.mean(self.r20)}")
        print(f"Validation NDCG@1: {np.mean(self.n1)}")
        print(f"Validation NDCG@3: {np.mean(self.n3)}")
        print(f"Validation NDCG@5: {np.mean(self.n5)}")
        print(f"Validation PSPrecision@1: {np.mean(self.psp1)}")
        print(f"Validation PSPrecision@3: {np.mean(self.psp3)}")
        print(f"Validation PSPrecision@5: {np.mean(self.psp5)}")
        print(f"Validation PSNDCG@1: {np.mean(self.psn1)}")
        print(f"Validation PSNDCG@3: {np.mean(self.psn3)}")
        print(f"Validation PSNDCG@5: {np.mean(self.psn5)}")

        # reset metrics
        self.p1 = []
        self.p3 = []
        self.p5 = []
        self.r5 = []
        self.r10 = []
        self.r20 = []
        self.n1 = []
        self.n3 = []
        self.n5 = []
        self.psp1 = []
        self.psp3 = []
        self.psp5 = []
        self.psn1 = []
        self.psn3 = []
        self.psn5 = []

    def on_test_epoch_end(
        self,
    ):
        self.log("Precision@1", np.mean(self.test_p1), sync_dist=True)
        self.log("Precision@3", np.mean(self.test_p3), sync_dist=True)
        self.log("Precision@5", np.mean(self.test_p5), sync_dist=True)

        self.log("Recall@5", np.mean(self.test_r5), sync_dist=True)
        self.log("Recall@10", np.mean(self.test_r10), sync_dist=True)
        self.log("Recall@20", np.mean(self.test_r20), sync_dist=True)

        self.log("NDCG@1", np.mean(self.test_n1), sync_dist=True)
        self.log("NDCG@3", np.mean(self.test_n3), sync_dist=True)
        self.log("NDCG@5", np.mean(self.test_n5), sync_dist=True)

        self.log("PSPrecision@1", np.mean(self.test_psp1), sync_dist=True)
        self.log("PSPrecision@3", np.mean(self.test_psp3), sync_dist=True)
        self.log("PSPrecision@5", np.mean(self.test_psp5), sync_dist=True)

        self.log("PSNDCG@1", np.mean(self.test_psn1), sync_dist=True)
        self.log("PSNDCG@3", np.mean(self.test_psn3), sync_dist=True)
        self.log("PSNDCG@5", np.mean(self.test_psn5), sync_dist=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            model=self,
            optimizer_name=self.hparams.optimizer,
            learning_rate=self.hparams.learning_rate,
            eps=self.hparams.eps,
            betas=[self.hparams.beta1, self.hparams.beta2],
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = get_scheduler(
            scheduler_name=self.hparams.scheduler,
            optimizer=optimizer,
            max_learning_rate=self.hparams.learning_rate,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=self.hparams.num_cycles,
            pct_start=self.hparams.pct_start,
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "lr",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _freeze_encoders(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.label_encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoders(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.label_encoder.parameters():
            param.requires_grad = True

    def _initialize_metrics(self):
        # precision
        self.p1, self.p3, self.p5 = [], [], []
        self.test_p1, self.test_p3, self.test_p5 = [], [], []
        # recall
        self.r5, self.r10, self.r20 = [], [], []
        self.test_r5, self.test_r10, self.test_r20 = [], [], []
        # ndcg
        self.n1, self.n3, self.n5 = [], [], []
        self.test_n1, self.test_n3, self.test_n5 = [], [], []
        # psp
        self.psp1, self.psp3, self.psp5 = [], [], []
        self.test_psp1, self.test_psp3, self.test_psp5 = [], [], []
        # psndcg
        self.psn1, self.psn3, self.psn5 = [], [], []
        self.test_psn1, self.test_psn3, self.test_psn5 = [], [], []

    def _calc_metrics(self, y_hat, y, propensities, test=False):
        y_hat = y_hat.detach().cpu().numpy()
        y = sparse.csr_matrix(y.detach().cpu().numpy())

        p = precision(y_hat, y, k=5, sorted=False)
        r = recall(y_hat, y, k=20, sorted=False)
        n = ndcg(y_hat, y, k=5, sorted=False)
        psp = psprecision(y_hat, y, inv_psp=propensities, k=5, sorted=False)
        psn = psndcg(y_hat, y, inv_psp=propensities, k=5, sorted=False)

        if not test:
            self.p1.append(p[0])
            self.p3.append(p[2])
            self.p5.append(p[4])

            self.r5.append(r[4])
            self.r10.append(r[9])
            self.r20.append(r[19])

            self.n1.append(n[0])
            self.n3.append(n[2])
            self.n5.append(n[4])

            self.psp1.append(psp[0])
            self.psp3.append(psp[2])
            self.psp5.append(psp[4])

            self.psn1.append(psn[0])
            self.psn3.append(psn[2])
            self.psn5.append(psn[4])

        if test:
            self.test_p1.append(p[0])
            self.test_p3.append(p[2])
            self.test_p5.append(p[4])

            # test recall@k
            self.test_r5.append(r[4])
            self.test_r10.append(r[9])
            self.test_r20.append(r[19])

            # test ndcg@k
            self.test_n1.append(n[0])
            self.test_n3.append(n[2])
            self.test_n5.append(n[4])

            # test psprecision@k
            self.test_psp1.append(psp[0])
            self.test_psp3.append(psp[2])
            self.test_psp5.append(psp[4])

            # test psndcg@k
            self.test_psn1.append(psn[0])
            self.test_psn3.append(psn[2])
            self.test_psn5.append(psn[4])

    def _calc_loss(self, y_hat, y, propensities):
        if self.hparams.propensity_loss:
            BCE_fn = MaskedBCEWithLogitsLoss(
                reduction="none",
                weight=torch.tensor(propensities).to(self.device),
                n_mask=self.current_n_mask,
            )
        else:
            BCE_fn = MaskedBCEWithLogitsLoss(
                reduction="none",
                n_mask=self.current_n_mask,
            )

        CE_fn = MultiLabelCrossEntropyLoss()
        if self.hparams.logit_scaling:
            y_hat = scale_tensor(y_hat, min_val=-6, max_val=6, dim=1)

        if self.hparams.loss_fn == "mix":
            if self.hparams.propensity_loss:
                CE_loss = CE_fn(y_hat.float(), y.float(), weights=propensities)
            else:
                CE_loss = CE_fn(y_hat.float(), y.float(), weights=None)

            BCE_loss = BCE_fn(y_hat.float(), y.float())
            loss = self.hparams.alpha * CE_loss + (1 - self.hparams.alpha) * BCE_loss

        if self.hparams.loss_fn == "CE":
            if self.hparams.propensity_loss:
                loss = CE_fn(y_hat.float(), y.float(), weights=propensities)
            else:
                loss = CE_fn(y_hat.float(), y.float(), weights=None)

        if self.hparams.loss_fn == "BCE":
            loss = BCE_fn(y_hat.float(), y.float())

        return loss

    def compute_inner_product(self, doc_embeddings, label_embeddings):
        # Reshape doc_embeddings from [128, 768] to [128, 1, 768]
        doc_embeddings = doc_embeddings.unsqueeze(1)

        # Perform batched matrix multiplication
        # The result will be of shape [128, 1, 1000], so we squeeze the second dimension
        inner_product = torch.bmm(
            doc_embeddings, label_embeddings.transpose(1, 2)
        ).squeeze(1)
        return inner_product

    def _pool(self, model_output, attention_mask, strategy="none"):
        assert strategy in [
            "none",
            "mean",
            "concat",
        ], f"Pooling strategy {strategy} not supported"

        if strategy == "none":
            return model_output[0][:, 0, :]

        # https://huggingface.co/efederici/sentence-bert-base
        if strategy == "mean":
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        if strategy == "concat":
            # get 5 last hidden states and concatenate them
            last_hidden_states = model_output["hidden_states"]
            last_5_cls = [
                hidden_layer[:, 0, :] for hidden_layer in last_hidden_states[-5:]
            ]
            return self.projection_layer(torch.concat(last_5_cls, dim=1))


class Head(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 8),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


class SkipConnectionHead(nn.Module):
    def __init__(self, transformer_output_dim, dropout=0.0, skip_dropout=0.0):
        super().__init__()
        self.gates = nn.Parameter(torch.Tensor(transformer_output_dim))
        self.reset_parameters()
        self.skip_dropout_prob = skip_dropout
        self.dropout = nn.Dropout(p=dropout)

        self.skip_dropout = (
            nn.Dropout(p=self.skip_dropout_prob)
            if self.skip_dropout_prob > 0.0
            else nn.Identity()
        )
        self.fc = nn.Linear(transformer_output_dim, transformer_output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(self.skip_dropout(x)) * torch.sigmoid(self.gates) + x

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(
            self.gates, -4.0
        )  # -4. ensures that all vector components are disabled by default
