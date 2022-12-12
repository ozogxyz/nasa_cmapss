from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import rul_datasets

PYTORCH_ENABLE_MPS_FALLBACK = 1

# Check first that CUDA then MPS is available, if not fallback to CPU
if torch.cuda.is_available():
    device = "cuda:0"
elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    device = torch.device("mps")


class Cnn1dLSTM(LightningModule):
    def __init__(
        self,
        batch_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        maxpool_kernel: int,
        num_classes:  int,
        hidden_size: int,
        num_layers: int,
        maxpool_stride: Optional[int],
        window_size: Optional[int],
        learning_rate: int,
    ):
        """
        :param batch_size: Input batch size.
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.maxpool_kernel = maxpool_kernel
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.maxpool_stride = maxpool_stride
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        # CNN feature extraction part
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_2 = nn.Conv1d(out_channels, out_channels*2, kernel_size-2)
        self.maxpool = nn.MaxPool1d(
            kernel_size=maxpool_kernel, stride=maxpool_stride)

        conv_out_dim = window_size - kernel_size - 1
        maxpool_out_dim = (conv_out_dim - maxpool_kernel) // maxpool_stride + 1
        embedding_length = maxpool_out_dim

        # LSTM Part
        self.lstm = nn.LSTM(embedding_length, hidden_size,
                            num_layers, dropout=0.2)

        # Regressor part
        self.fc_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, time_series):
        """
        rul-datasets library gives batches of:
        n_batch x n_features(14) x window_size(30)

        :param time_series: tensor([batch_size, 14, 30])
        :return: predictions: tensor([batch_size])
        """

        # CNN feature extraction
        features = torch.relu(self.conv_1(time_series))
        features = torch.relu(self.conv_2(features))

        # Max-pooling layer
        features = self.maxpool(features)

        lstm_output, _ = self.lstm(features)
        lstm_output = torch.tanh(lstm_output)
        lstm_output = lstm_output[:, -1]

        # Regressor
        output = self.fc_1(lstm_output)
        output = torch.relu(output)
        output = self.fc_2(output)
        return output.flatten()

    def training_step(self, batch):
        time_series, labels = batch
        predictions = self.forward(time_series)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, labels)
        loss = torch.sqrt(loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self.forward(features)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, labels)
        loss = torch.sqrt(loss)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, dataloader_idx=1):
        features, labels = batch
        predictions = self.forward(features)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, labels)
        loss = torch.sqrt(loss)
        self.log(f'test_loss: ', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return super().train_dataloader()
    
    def val_dataloader(self):
        return super().val_dataloader()