from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from model import ExectModel
from datamodule import ExectDataModule
from callbacks import EmbeddingCallback, SaveSparsePredictionCallback

# fix to many files open
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision("medium") # to make lightning happy

class ExectLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EmbeddingCallback, "embedding_callback")
        parser.add_lightning_class_args(SaveSparsePredictionCallback, "sparse_prediction_callback")
        parser.link_arguments("model.model", "data.model", apply_on="instantiate")


def cli_main():
    cli = ExectLightningCLI(
        ExectModel,
        ExectDataModule,
        seed_everything_default=1337,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
