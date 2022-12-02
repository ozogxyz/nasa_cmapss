import warnings

import numpy as np
import pytorch_lightning as pl
import rul_datasets

from models.models import *

pl.seed_everything(42)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

if __name__ == "__main__":
    # read data
    cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
    dm = rul_datasets.RulDataModule(
        cmapss_fd1,
        batch_size=30,
    )
    # instantiate model
    cnn_lstm = Shcherbakov(
        in_channels=14, 
        out_channels=14, 
        hidden_channels=30, 
        kernel_size=30
    )
    # create trainer context
    trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        max_epochs=100
    )
    # fit & test
    trainer.fit(cnn_lstm, dm)
    trainer.test(cnn_lstm, dm)
