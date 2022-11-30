import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from pytorch_lightning import LightningModule


class LSTM(LightningModule):
    def __init__(self, input_size, hidden_size, seq_length=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.1
        )
        self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
        self.fc_2 = nn.Linear(16, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_size))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn_o = torch.Tensor(hn.detach().numpy()[-1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size)
        hn_1 = torch.Tensor(hn.detach().numpy()[1, :, :])
        hn_1 = hn_1.view(-1, self.hidden_size)
        out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = torch.nn.L1Loss(self.x, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
