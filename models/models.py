import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F


class Shcherbakov(LightningModule):
    def __init__(self, dim_in, dim_lstm, kernel_size):
        # n_channels = num_sensors = 14
        super().__init__()
        self.network = nn.Sequential()
        self.network.add_module('conv1', nn.Conv1d(dim_in, dim_in * 2, kernel_size))
        self.network.add_module('relu_conv1', nn.ReLU())
        self.network.add_module('conv2', nn.Conv1d(dim_in * 2, dim_lstm, kernel_size))
        self.network.add_module('relu_conv2', nn.ReLU())
        self.network.add_module('max_pool', nn.MaxPool1d(kernel_size))
        self.network.add_module('relu_max_pool', nn.ReLU())
        self.network.add_module('lstm1', nn.LSTM(dim_lstm, dim_lstm))
        self.network.add_module('tanh_lstm1', nn.Tanh())
        self.network.add_module('drop1', nn.Dropout(p=0.2))
        self.network.add_module('tanh_drop1', nn.Tanh())
        self.network.add_module('lstm2', nn.LSTM(dim_lstm, dim_lstm))
        self.network.add_module('tanh_lstm2', nn.Tanh())
        self.network.add_module('drop2', nn.Dropout(p=0.2))
        self.network.add_module('tanh_drop2', nn.Tanh())
        self.network.add_module('flatten', nn.Flatten())
        self.network.add_module('dense', nn.Linear(dim_lstm, dim_lstm // 2))
        self.network.add_module('relu_dense', nn.ReLU())
        self.network.add_module('output', nn.Linear(dim_lstm // 2, 1))

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch):
        x, y = batch
        loss = F.l1_loss(x, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        val_loss = F.l1_loss(x, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        test_loss = F.l1_loss(x, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer


class Net(LightningModule):
    def __init__(self, dim_in, dim_lstm, kernel_size):
        # n_channels = num_sensors = 14
        super().__init__()
        # self.network = nn.Sequential()
        self.layer1 = nn.Conv1d(dim_in, dim_in * 2, kernel_size)
        self.layer2 = nn.Conv1d(dim_in * 2, dim_lstm, kernel_size)
        self.layer3 = nn.MaxPool1d(kernel_size)
        self.layer4 = nn.LSTM(dim_lstm, dim_lstm)
        self.layer5 = nn.LSTM(dim_lstm, dim_lstm)
        self.layer6 = nn.Flatten()
        self.layer7 = nn.Linear(dim_lstm, dim_lstm // 2)
        self.layer8 = nn.Linear(dim_lstm // 2, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(F.max_pool1d(x))
        x = F.dropout(F.tanh(self.layer4(x)), p=0.2)
        x = F.dropout(F.tanh(self.layer5(x)), p=0.2)
        x = self.layer6(x)
        x = F.relu(self.layer7(x))
        x = self.layer8(x)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        loss = F.l1_loss(x, y)
        return loss

    def validation_step(self, batch, batch_index):
        # this is the validation loop
        x, y = batch
        val_loss = F.l1_loss(x, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_index):
        # this is the test loop
        x, y = batch
        test_loss = F.l1_loss(x, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
