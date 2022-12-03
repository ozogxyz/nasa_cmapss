import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Network(LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # CNN feature extraction part
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_2 = nn.Conv1d(out_channels, out_channels * 2, kernel_size - 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.num_layers = 50
        self.lstm_1 = nn.LSTM(input_size=8, hidden_size=16, num_layers=self.num_layers)
        self.lstm_2 = nn.LSTM(input_size=16, hidden_size=32, num_layers=self.num_layers)
        self.dropout = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()
        self.regressor = nn.Linear(2048, 1)

    def forward(self, time_series):
        # CNN feature extraction
        interim = self.relu(self.conv_1(time_series))
        features = self.relu(self.conv_2(interim))
        features = self.maxpool(features)

        # LSTM layers
        h_0 = torch.zeros(self.num_layers, features.size(1), 16).to("mps:0")
        c_0 = torch.zeros(self.num_layers, features.size(1), 16).to("mps:0")
        output, (h_n, c_n) = self.lstm_1(features, (h_0, c_0))
        output = self.tanh(self.dropout(output))

        h_0 = torch.zeros(self.num_layers, output.size(1), output.size(2) * 2).to("mps:0")
        c_0 = torch.zeros(self.num_layers, output.size(1), output.size(2) * 2).to("mps:0")
        output, (h_n, c_n) = self.lstm_2(output, (h_0, c_0))
        output = self.tanh(self.dropout(output))

        # Regressor
        output = self.flatten(output)
        output = self.dropout(self.relu(self.regressor(output)))
        return output

    def training_step(self, batch) -> STEP_OUTPUT:
        time_series, labels = batch
        preds = self.forward(time_series)
        loss = nn.functional.mse_loss(preds, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds, labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds, labels)
        self.log(f"test_loss_fd{dataloader_idx + 1}", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer
