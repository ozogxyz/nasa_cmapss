import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, predictions, labels):
        loss = torch.sqrt(self.mse(predictions, labels) + self.eps)
        return loss


class Shcherbakov(LightningModule):
    def __init__(self, in_channels=30, out_channels=32, kernel_size=5, num_layers=2, hidden_size=50):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # CNN feature extraction part
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, kernel_size - 2),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # LSTM Part
        self.lstm1 = nn.LSTM(
            input_size=4,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        )

        # Regressor part
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 1),
        )

    def forward(self, time_series):
        """
        rul-datasets library gives batches of:
        n_batch x n_features(14) x window_size(30)

        :param time_series: tensor([batch_size, 14, 30])
        :return: predictions: tensor([batch_size, 1])
        """
        # Swap window size and feature columns for convolution
        time_series = torch.moveaxis(time_series, 2, 1)

        # CNN feature extraction
        features = self.conv_1(time_series)
        features = self.conv_2(features)
        features = self.maxpool(features)
        # features = torch.flatten(features, start_dim=1)

        # LSTM layers
        h_0 = torch.zeros(self.num_layers, features.size(1), self.hidden_size).to("mps")
        c_0 = torch.zeros(self.num_layers, features.size(1), self.hidden_size).to("mps")
        output, (h_n, c_n) = self.lstm1(features, (h_0, c_0))
        output = torch.tanh(output)
        output, (h_n, c_n) = self.lstm2(output, (h_n, c_n))
        output = torch.tanh(output)

        # Regressor
        output = self.regressor(output)
        return output

    def training_step(self, batch):
        time_series, labels = batch
        predictions = self.forward(time_series)
        loss = RMSELoss().forward(predictions=predictions, labels=labels)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer
