import rul_datasets
import warnings
import pytorch_lightning as pl
import torch
from models.models import Network
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(42)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

if __name__ == "__main__":
    # read data
    model = Network(in_channels=30, out_channels=32, kernel_size=5)
    dev_features, dev_targets = rul_datasets.CmapssReader(fd=1).load_split("dev")
    time_series = torch.tensor(dev_features[0], dtype=torch.float)
    preds = model.forward(time_series).detach()
    targets = torch.tensor(dev_targets[0])
    print(preds)

