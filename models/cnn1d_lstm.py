import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from typing import Union, List, Tuple, Dict, Optional


PYTORCH_ENABLE_MPS_FALLBACK = 1


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, predictions, labels):
        loss = torch.sqrt(self.mse(predictions.flatten(), labels) + self.eps)
        return loss


class Cnn1dLSTM(LightningModule):
    def __init__(
        self,
        batch_size: int = 200,
        in_channels: int = 14,
        out_channels: int = 32,
        kernel_size: int = 5,
        maxpool_kernel: int = 3,
        num_classes:  int = 1,
        input_size: int = 14,
        hidden_size: int = 50,
        num_layers: int = 2,
        maxpool_stride: Optional[int] = 2,
        window_size: Optional[int] = 30,
        flatten: Optional[bool] = False,
    ):
        """
        :param batch_size: Input batch size.
        :param in_channels: Input channels for Conv1d, number of features (14 in FD001).
        :param out_channels: Output channels for 1st Conv1d, total conv output will be 2 * out_channels.
        :param kernel_size: Kernel size for 1st Conv1d, will be kernel_size - 2 in the next later.
        :param maxpool_kernel: Kernel size for pooling layer.
        :param num_classes:
        :param input_size: Sequence length for the LSTM layer. Equals to 2 * out_channels,
               which is the output of the convolutional layers.
        :param hidden_size: Size of the hidden state in LSTM layers.
        :param num_layers: Determines how many LSTM layers will be stacked.
        :param maxpool_stride: Pooling stride.

        """
        super().__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.maxpool_kernel = maxpool_kernel
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.maxpool_stride = maxpool_stride
        self.flatten = flatten

        # CNN feature extraction part
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size-2)
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_kernel, stride=maxpool_stride)

        conv_out_dim = window_size - kernel_size - 1
        seq_length = out_channels
        maxpool_out_dim = (conv_out_dim - maxpool_kernel) // maxpool_stride + 1
        if flatten:
            embedding_length = maxpool_out_dim * out_channels
        else:
            embedding_length = maxpool_out_dim

        # LSTM Part
        self.lstm = nn.LSTM(embedding_length, hidden_size)

        # Regressor part
        self.fc_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_2 = nn.Linear(hidden_size // 2, num_classes)

        # Activators etc
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer

    def forward(self, time_series):
        """
        rul-datasets library gives batches of:
        n_batch x n_features(14) x window_size(30)

        :param time_series: tensor([batch_size, 14, 30])
        :return: predictions: tensor([batch_size, 1])
        """

        # CNN feature extraction
        features = self.relu(self.conv_1(time_series))
        features = self.relu(self.conv_2(features))

        # Max-pooling layer
        features = self.maxpool(features)
        if self.flatten:
            features = torch.flatten(features, start_dim=1)

        # LSTM layers input: tensor of shape (L, N, H_in) for un-batched input, (L, N, H_in) when batch_first=False
        # or (N, L, Hin) when batch_first=True. Here features is of size (N=200, L=64, H_in=maxpool_out). h_0: tensor
        # of shape (D*num_layers, H_out) for un-batched input or (D*num_layers, N, H_out) containing the initial hidden
        # state for each element in the input sequence. Defaults to zeros if (h_0, c_0) is not provided.
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to("mps")
        c_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to("mps")

        features = torch.reshape(features, (features.size(0), 1, features.size(1)))
        lstm_output, hidden_state = self.lstm(features)
        lstm_output = self.tanh(lstm_output)

        # Regressor
        if not self.flatten:
            lstm_output = lstm_output[:, -1]
        output = self.relu(self.fc_1(lstm_output))
        output = self.fc_2(output)
        return output

    def training_step(self, batch):
        time_series, labels = batch
        predictions = self.forward(time_series)
        loss = RMSELoss().forward(predictions=predictions, labels=labels)
        # criterion = nn.MSELoss()
        # loss = criterion(predictions.flatten(), labels)
        # loss = RMSELoss().forward(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self.forward(features)
        loss = RMSELoss().forward(predictions=predictions, labels=labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, dataloader_idx=1):
        features, labels = batch
        predictions = self.forward(features)
        loss = RMSELoss().forward(predictions=predictions, labels=labels)
        self.log(f'RMSE for dataset FD00{dataloader_idx}.txt', loss)

