import warnings

import pytorch_lightning as pl
import rul_datasets
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import TensorBoardLogger

from models.network import Network
from utils.callbacks import PrintCallback
from utils.read_params import read_params

pl.seed_everything(42)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

PARAMS_FILEPATH = 'params.yaml'

if __name__ == "__main__":
    ####################################
    # 0. Read Hyper-Parameters
    fd, batch_size, window_size, in_channels, out_channels, kernel_size, stride, maxpool_kernel, maxpool_stride, num_classes, hidden_size, num_layers, max_epochs, patience, min_delta, lr, dropout = read_params(PARAMS_FILEPATH)
    ####################################
    
    ####################################
    # 1. Create Model
    ####################################
    model = Network(
        window_size=window_size,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        maxpool_kernel=maxpool_kernel,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        maxpool_stride=maxpool_stride,
        lr=lr,
        dropout=dropout,
    )

    ####################################
    # 2. Read Data
    ####################################
    cmapss_fd1 = rul_datasets.CmapssReader(fd)
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=batch_size)

    ####################################
    # 3. Callbacks
    ####################################
    early_stop_callback = EarlyStopping(
        check_on_train_epoch_end=True,
        monitor="val_loss",
        patience=patience,
        strict=False,
        verbose=True,
        min_delta=min_delta,
        mode='min'
    )
    learning_rate_monitor = LearningRateMonitor()
    model_summary = RichModelSummary()
    print_callback = PrintCallback()
    progress_bar = RichProgressBar()
    timer = Timer()

    ####################################
    # 4. Trainer
    ####################################
    logger = TensorBoardLogger(save_dir='tmp', name='tb_logs')
    trainer = pl.Trainer(
        auto_lr_find=True,
        accelerator='auto',
        callbacks=[early_stop_callback, learning_rate_monitor ,model_summary,
            print_callback, progress_bar, timer],
        log_every_n_steps=15,
        max_epochs=max_epochs,
        logger=logger,
        default_root_dir='tmp/checkpoints/'
    )

    ####################################
    # 5. Hyper-parameter Tuning
    ####################################

    #####################################
    # 6. Fit
    #####################################
    trainer.fit(model, dm)
    trainer.test(model, dm)
    