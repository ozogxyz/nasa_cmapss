from typing import Any

import torch.nn as nn


class Conv1DBlock(nn.Module):
    """
    1D convolutional block
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super(Conv1DBlock, self).__init__()
        """
        Initialize Conv1DBlock
        :param args: arg.blocks.conv?.yaml (see hydra configs).
        """
        self.conv1d = args[0]
        self.bn1d = args[1]
        self.activation = args[2]
        self.dropout = args[3]

    def forward(self, x):
        """
        :param x: Input tensor
        :return: Output tensor
        """
        x = self.conv1d(x)
        x = self.bn1d(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
            return x
        return x


class LSTMBlock(nn.Module):
    """
    Long short-term memory (LSTM) block
    """

    def __init__(self, *args: Any, **kwargs):
        super(LSTMBlock, self).__init__()
        """
        Initialize LSTMBlock
        :param args: arg.blocks.lstm?.yaml (see hydra configs).
        """
        self.lstm = args[0]
        self.activation = args[1]
        self.dropout = args[2]

    def forward(self, x):
        """
        :param x: Input tensor
        :return: Output tensor
        """
        x, _ = self.lstm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
            return x
        return x


class Regressor(nn.Module):
    """
    Regressor
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super(Regressor, self).__init__()
        """
        Initialize Regressor
        :param args: arg.blocks.regressor?.yaml (see hydra configs).
        """
        self.fc1 = args[0]
        self.activation1 = args[1]
        self.dropout1 = args[2]
        self.fc2 = args[3]
        self.activation2 = args[4]
        self.dropout2 = args[5]
        self.fc3 = args[6]


    def forward(self, x):
        """
        :param x: Input tensor
        :return: Output tensor
        """
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class MaxPool(nn.Module):
    """
    Maxpool
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super(MaxPool, self).__init__()
        """
        Initialize Maxpool
        :param args: arg.blocks.maxpool?.yaml (see hydra configs).
        """
        self.pool = args[0]

    def forward(self, x):
        """
        :param x: Input tensor
        :return: Output tensor
        """
        x = self.pool(x)
        return x
