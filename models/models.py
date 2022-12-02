import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F


class Shcherbakov(LightningModule):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size):
        # n_channels = num_sensors = 14
        super().__init__()
        self.layer1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.layer2 = nn.Conv1d(out_channels, hidden_channels, hidden_channels)
        self.layer3 = nn.MaxPool1d(hidden_channels)
        self.layer4 = nn.LSTM(hidden_channels, hidden_channels)
        self.layer5 = nn.LSTM(hidden_channels, hidden_channels)
        self.layer6 = nn.Flatten()
        self.layer7 = nn.Linear(hidden_channels, hidden_channels)
        self.layer8 = nn.Linear(hidden_channels , out_channels)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.dropout(F.tanh(self.layer4(x)), p=0.2)
        x = F.dropout(F.tanh(self.layer5(x)), p=0.2)
        x = self.layer6(x)
        x = F.relu(self.layer7(x))
        x = self.layer8(x)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        # x = self.forward(x)
        loss = F.l1_loss(x, y)
        return loss

    def validation_step(self, batch, batch_index):
        # this is the validation loop
        x, y = batch
        x = self.forward(x)
        val_loss = F.l1_loss(x, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_index):
        # this is the test loop
        x, y = batch
        x = self.forward(x)
        test_loss = F.l1_loss(x, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
