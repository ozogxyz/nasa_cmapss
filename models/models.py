import torch
import torch.nn as nn

from pytorch_lightning import LightningModule


class Shcherbakov(LightningModule):
    """CNN_LSTM network as per paper by Shcherbakov and Sai.

    Paper: N_L: Window size x N_F: Number of features
            In FD001: N_L=30 N_F=14.
            Number of filters? : 16:16:256
            Kernel size for Conv1D: 1:1:10
            LSTM neurons: 10:10:100
            Dropout rate: 0.0:0.1:0.5

    Input to Conv1D:
                    batch_size x in_channels x out_channels?
                    32         x    14       x      32

                nn.Conv1D(14, ? , kernel_size)
    Number of filters = number of channels
    """

    def __init__(self, in_channels, out_channels, hidden_size, kernel_size):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Flatten(),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )
        self.lstm_1 = nn.LSTM(829, hidden_size=hidden_size)
        self.lstm_2 = nn.LSTM(hidden_size, 50)
        self.regressor = nn.Sequential(
            nn.Linear(829, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )

    def forward(self, time_series):
        features = self.feature_extractor(time_series)
        # h_0 = Variable(torch.zeros(features.size(0), 50))
        # c_0 = Variable(torch.zeros(features.size(0), 50))
        # output, (h_n, c_n) = self.lstm_1(h_0, c_0)
        # h_n = F.dropout(F.tanh(h_n), p=0.2)
        # c_n = F.dropout(F.tanh(c_n), p=0.2)
        # output, (h_n, c_n) = self.lstm_1(h_n, c_n)
        return self.regressor(features)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds / 50, labels / 50)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds, labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds, labels)
        self.log(f"test_loss_fd{dataloader_idx + 1}", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer


class Test(LightningModule):
    """CNN_LSTM network as per paper by Shcherbakov and Sai.

    Paper: N_L: Window size x N_F: Number of features
            In FD001: N_L=30 N_F=14.
            Number of filters? : 16:16:256
            Kernel size for Conv1D: 1:1:10
            LSTM neurons: 10:10:100
            Dropout rate: 0.0:0.1:0.5

    Input to Conv1D:
                    batch_size x in_channels x out_channels?
                    32         x    14       x      32

                nn.Conv1D(14, ? , kernel_size)
    Number of filters = number of channels
    """

    def __init__(self, in_channels, out_channels, hidden_size, kernel_size):
        super().__init__()
        # CNN feature extraction part
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('conv1', nn.Conv1d(in_channels[0], out_channels[0], kernel_size[0]))
        self.feature_extractor.add_module('relu_conv1', nn.ReLU())
        self.feature_extractor.add_module('conv1', nn.Conv1d(in_channels[0], out_channels[0], kernel_size[0]))
        self.feature_extractor.add_module('relu_conv2', nn.ReLU())
        self.feature_extractor.add_module('maxpool', nn.MaxPool1d(kernel_size=2, stride=1))
        self.feature_extractor.add_module('relu_max_pool', nn.ReLU())

        # LSTM part
        self.recurrent = nn.Sequential()
        self.recurrent.add_module('lstm1', nn.LSTM(829, hidden_size=hidden_size))
        self.recurrent.add_module('tanh_lstm1', nn.Tanh())
        self.recurrent.add_module('drop1', nn.Dropout(p=0.2))
        self.recurrent.add_module('lstm2', nn.LSTM(hidden_size, 50))
        self.recurrent.add_module('tanh_lstm1', nn.Tanh())
        self.recurrent.add_module('drop1', nn.Dropout(p=0.2))

        # Regressor
        self.regressor = nn.Sequential()
        self.regressor.add_module('dense', nn.Linear(829, 25))
        self.regressor.add_module('relu_dense', nn.ReLU())
        self.regressor.add_module('output', nn.Linear(25, 1))
        self.regressor.add_module('softmax', nn.Softmax())

    def forward(self, time_series):
        features = self.feature_extractor(time_series)
        print("")
        # h_0 = Variable(torch.zeros(features.size(0), 50))
        # c_0 = Variable(torch.zeros(features.size(0), 50))
        # output, (h_n, c_n) = self.lstm_1(h_0, c_0)
        # h_n = F.dropout(F.tanh(h_n), p=0.2)
        # c_n = F.dropout(F.tanh(c_n), p=0.2)
        # output, (h_n, c_n) = self.lstm_1(h_n, c_n)
        return self.regressor(features)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds / 50, labels / 50)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds, labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds, labels)
        self.log(f"test_loss_fd{dataloader_idx + 1}", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
