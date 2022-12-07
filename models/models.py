import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, predictions, labels):
        loss = torch.sqrt(self.mse(predictions.flatten(), labels) + self.eps)
        return loss


class Shcherbakov(LightningModule):
    def __init__(self, in_channels=14, out_channels=32, kernel_size=5, num_layers=2, hidden_size=96):
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
        self.maxpool = nn.MaxPool1d(kernel_size=3)

        # LSTM Part
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.2)

        # Regressor part
        self.regressor = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.ReLU(),
        )

    def forward(self, time_series):
        # CNN feature extraction
        features = self.conv_1(time_series)
        features = self.conv_2(features)
        features = self.maxpool(features)
        features = torch.flatten(features, start_dim=1)

        # LSTM layers
        h_0 = torch.zeros(self.num_layers, self.hidden_size).to("mps")
        c_0 = torch.zeros(self.num_layers, self.hidden_size).to("mps")
        # h_0 = torch.zeros(self.num_layers, self.hidden_size)
        # c_0 = torch.zeros(self.num_layers, self.hidden_size)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
