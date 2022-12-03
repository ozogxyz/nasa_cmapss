import rul_datasets
import torch
import warnings

import pytorch_lightning as pl

from models.models import Network
pl.seed_everything(42)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

if __name__ == "__main__":
    # read data
    cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
    # create model
    model = Network(in_channels=14, out_channels=32, kernel_size=5)
    # create trainer context
    trainer = pl.Trainer(accelerator="mps", devices=1, max_epochs=10)
    # fit & test
    trainer.fit(model, dm)
    trainer.test(model, dm)
