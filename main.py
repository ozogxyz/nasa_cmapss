import rul_datasets
import warnings
import pytorch_lightning as pl
from models.models import Shcherbakov
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(42)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


if __name__ == "__main__":
    # read data
    cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=200)
    # create model
    model = Shcherbakov()
    # create trainer context
    logger = TensorBoardLogger("tb_logs", name="Shcherbakov")
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=100, logger=logger)
    # fit & test
    trainer.fit(model, dm)
    trainer.test(model, dm)
