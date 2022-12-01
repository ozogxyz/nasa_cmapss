import pytorch_lightning as pl
import rul_datasets
from rul_datasets import CmapssReader

from models import rul_estimator

# import rul_estimator # for fd in [1, 2, 3, 4]:

if __name__ == "__main__":
    for fd in [1]:
        cmapss: CmapssReader = rul_datasets.CmapssReader(fd)
        dm = rul_datasets.RulDataModule(cmapss, batch_size=32)

        my_rul_estimator = rul_estimator.LSTM(16, 96, 4)

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=100,
        )
        print(dm)
        trainer.fit(my_rul_estimator, dm)

        trainer.test(my_rul_estimator, dm)
