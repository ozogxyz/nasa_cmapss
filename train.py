import os
import warnings
import torch
import pytorch_lightning as pl
import rul_datasets
import yaml
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import TensorBoardLogger
from utils.callbacks import PrintCallback
from models.cnn1d_lstm import Cnn1dLSTM
from pprint import pprint

pl.seed_everything(42)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


if __name__ == "__main__":
    # read hyperparameters
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    # dataset params
    fd = params.get('dataset').get('filename')
    batch_size = params.get('dataset').get('batch_size')

    # model params
    in_channels = params.get('model').get('in_channels')
    out_channels = params.get('model').get('out_channels')
    kernel_size = params.get('model').get('kernel_size')
    maxpool_kernel = params.get('model').get('maxpool_kernel')
    num_classes = params.get('model').get('num_classes')
    hidden_size = params.get('model').get('hidden_size')
    num_layers = params.get('model').get('num_layers')
    maxpool_stride = params.get('model').get('maxpool_stride')
    window_size = params.get('model').get('window_size')

    # training params
    lr = params.get('training').get('lr')
    max_epochs = params.get('training').get('epochs')
    patience = params.get('training').get('patience')
    min_delta = params.get('training').get('min_delta')

    ####################################
    # 1. Create Model
    ####################################
    model = Cnn1dLSTM(
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        maxpool_kernel=maxpool_kernel,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        maxpool_stride=maxpool_stride,
        window_size=window_size,
        lr=lr,
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
        monitor="val_loss",
        patience=patience,
        strict=False,
        verbose=True,
        min_delta=min_delta,
        mode='min'
    )
    learning_rate_finder = LearningRateFinder()
    # learning_rate_monitor = LearningRateMonitor()
    model_summary = RichModelSummary()
    print_callback = PrintCallback()
    progress_bar = RichProgressBar()
    timer = Timer()

    ####################################
    # 4. Trainer
    ####################################
    logger = TensorBoardLogger(save_dir='tmp/tb_logs', log_graph=True)
    trainer = pl.Trainer(
        auto_lr_find=True,
        accelerator='auto',
        callbacks=[early_stop_callback, learning_rate_finder, model_summary,
                   print_callback, progress_bar, timer],
        gradient_clip_val=10,
        max_epochs=max_epochs,
    )

    ####################################
    # 5. Hyper-parameter Tuning
    ####################################
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model)

    # Results can be found in
    print(lr_finder.results)

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    #####################################
    # 5. Fit
    #####################################
    trainer.tune(model)
    trainer.fit(model, dm)
    trainer.test(model, dm)
