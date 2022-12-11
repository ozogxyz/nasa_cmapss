import os
import rul_datasets
import yaml
import warnings
import pytorch_lightning as pl
from models.cnn1d_lstm import Cnn1dLSTM
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

pl.seed_everything(42)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

if __name__ == "__main__":
    # read hyperparameters
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    batch_size = params.get('batch_size')
    fd = params.get('filename')
    max_epochs = params.get('epochs')
    in_channels = params.get('in_channels')
    out_channels = params.get('out_channels')
    kernel_size = params.get('kernel_size')
    maxpool_kernel = params.get('maxpool_kernel')
    num_classes = params.get('num_classes')
    hidden_size = params.get('hidden_size')
    num_layers = params.get('num_layers')
    maxpool_stride = params.get('maxpool_stride')
    window_size = params.get('window_size')
    lr = params.get('lr')

    # read data
    cmapss_fd1 = rul_datasets.CmapssReader(fd)
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=batch_size)
    # create model
    model = Cnn1dLSTM(
                batch_size = batch_size,
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                maxpool_kernel = maxpool_kernel,
                num_classes = num_classes,
                hidden_size = hidden_size,
                num_layers = num_layers,
                maxpool_stride = maxpool_stride,
                window_size = window_size,
                lr = lr,
    )

    # create trainer context
    early_stop_callback = EarlyStopping(monitor="train_loss", patience=3, min_delta=100, mode='min')
    logger = TensorBoardLogger("tb_logs", name="Cnn1dLSTM")
    trainer = pl.Trainer(callbacks=[early_stop_callback], accelerator='auto',max_epochs=max_epochs, logger=logger)

    os.system("bat params.yaml")

    # fit & test
    trainer.fit(model, dm)
    trainer.test(model, dm)
