from pprint import pprint
from typing import Optional

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule

from .cnn import Conv1D
from .linear import Linear
from .lstm import LSTM

PYTORCH_ENABLE_MPS_FALLBACK = 1

# Check first that CUDA then MPS is available, if not fallback to CPU
if torch.cuda.is_available():
    device = "cuda:0"
elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled   device on this machine.")
else:
    device = torch.device("mps")


class Network(LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            maxpool_kernel: int,
            num_classes: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            maxpool_stride: Optional[int],
            window_size: Optional[int],
            lr: Optional[int],
    ):
        """
        :param in_channels: Input channels for Conv1d, number of features (14 in FD001).
        :param out_channels: Output channels for 1st Conv1d, total conv output will be 2 * out_channels.
        :param kernel_size: Kernel size for 1st Conv1d, will be kernel_size - 2 in the next later.
        :param maxpool_kernel: Kernel size for pooling layer.
        :param num_classes:
        :param hidden_size: Size of the hidden state in LSTM layers.
        :param num_layers: Determines how many LSTM layers will be stacked.
        :param maxpool_stride: Pooling stride.

        """
        super().__init__()
        # Attributes
        self.lr = lr

        # Convolutional Blocks
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('Conv1D_1', Conv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
        self.feature_extractor.add_module('ELU', nn.ELU())
        self.feature_extractor.add_module('Conv1D_2', Conv1D(in_channels=out_channels, out_channels=out_channels+6, kernel_size=kernel_size-2, stride=stride))
        self.feature_extractor.add_module('ELU', nn.ELU())

        # Max-Pooling Layer
        self.maxpool = nn.Sequential()
        self.maxpool.add_module('MaxPool1D', nn.MaxPool1d(kernel_size=maxpool_kernel, stride=maxpool_stride))

        # LSTM Blocks
        conv1_out_dim = (window_size - kernel_size) // stride + 1
        conv2_out_dim = (conv1_out_dim - kernel_size+2) // stride + 1
        maxpool_out_dim = (conv2_out_dim - maxpool_kernel) // maxpool_stride + 1
        embedding_length = maxpool_out_dim

        self.temporal_extractor = nn.Sequential()
        self.temporal_extractor.add_module('LSTM', LSTM(embedding_length, hidden_size, num_layers, dropout))

        # Fully Connected Linear Layers for Regression
        self.regressor = nn.Sequential()
        self.regressor.add_module('FC_1', Linear(hidden_size, hidden_size//2))
        self.regressor.add_module('ReLU', nn.ReLU())
        self.regressor.add_module('FC_2', Linear(hidden_size//2, hidden_size//4))
        self.regressor.add_module('ReLU', nn.ReLU())
        self.regressor.add_module('FC_3', Linear(hidden_size//4, hidden_size//8))
        self.regressor.add_module('ReLU', nn.ReLU())
        self.regressor.add_module('FC_4', Linear(hidden_size//8, num_classes))

        # Metrics
        self.metric = torchmetrics.MeanSquaredError(squared=False)

        # Lightning Stuff
        self.save_hyperparameters()

    def forward(self, time_series):
        """
        rul-datasets library gives batches of:
        n_batch x n_features(14) x window_size(30)

        :param time_series: tensor([batch_size, 14, 30])
        :return: predictions: tensor([batch_size])
        """

        # Feature Extraction
        features = self.feature_extractor(time_series)

        # Max-Pooling
        features = self.maxpool(features)

        # Temporal Dependency Capture
        lstm_output, _ = self.temporal_extractor(features)
        lstm_output = lstm_output[:, -1]  # ? Neden hatırlamıyorum

        # Regression
        output = self.regressor(lstm_output)
        return output.flatten()

    def training_step(self, batch):
        time_series, target = batch
        preds = self.forward(time_series)
        loss = self.metric(preds, target)
        self.log_dict({'train_loss': self.metric})
        return loss

    def validation_step(self, batch, batch_idx):
        features, target = batch
        preds = self.forward(features)
        loss = self.metric(preds, target)
        self.log_dict({"val_loss": loss})

    def test_step(self, batch, dataloader_idx=1):
        features, target = batch
        preds = self.forward(features)
        loss = self.metric(preds, target)
        self.log('RMSE', value=loss, enable_graph=True, add_dataloader_idx=True, metric_attribute=self.metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_test_end(self) -> None:
        print(self.hparams)
        return super().on_test_end()
