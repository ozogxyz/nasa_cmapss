import logging
import warnings

import pytorch_lightning as pl
import rul_datasets

from models.models import *

pl.seed_everything(42)
logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def train():
    fd1 = rul_datasets.reader.CmapssReader(1)
    dm = rul_datasets.RulDataModule(fd1, batch_size=32)
    # dm = rul_datasets.BaselineDataModule(rul_datasets.RulDataModule(fd1, batch_size=32))
    lm = Test(
        in_channels=(14, 32),
        out_channels=(32, 64),
        hidden_size=50,
        kernel_size=(5, 3)
    )
    trainer = pl.Trainer(
        # accelerator='mps',
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=True,
    )
    trainer.fit(lm, dm)
    return trainer.test(lm, dm)


if __name__ == "__main__":
    test_results = train()