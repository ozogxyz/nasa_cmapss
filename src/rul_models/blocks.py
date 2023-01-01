from typing import Any
import torch
import torch.nn as nn
from abc import ABC, abstractclassmethod


class AbstractBlock(ABC):



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
        self.conv = nn.Sequential(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

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
        self.lstm = nn.Sequential(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lstm(x)


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
        self.regressor = nn.Sequential(*args, **kwargs)

    def forward(self, x):
        """
        :param x: Input tensor
        :return: Output tensor
        """
        return self.regressor(x)


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
        self.maxpool = nn.Sequential(*args, **kwargs)

    def forward(self, x):
        """
        :param x: Input tensor
        :return: Output tensor
        """
        return torch.max(x, self.maxpool(x))


class Linear(nn.Module):
    """
    Linear
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super(Linear, self).__init__()
        """
        Initialize Linear
        :param args: arg.blocks.linear?.yaml (see hydra configs).
        """
        self.linear = nn.Sequential(*args, **kwargs)

    def forward(self, x):
        """
        :param x: Input tensor
        :return: Output tensor
        """
        return self.linear(x)


class GRUBlock(nn.Module):
    """
    Gated Recurrent Unit (GRU) block
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super(GRUBlock, self).__init__()
        """
        Initialize GRUBlock
        :param args: arg.blocks.gru?.yaml (see hydra configs).
        """
        self.gru = nn.Sequential(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor
        :return: Output tensor
        """
        return self.gru(x)


class RNNBlock(nn.Module):
    """
    RNN Block
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super(RNNBlock, self).__init__()
        """
        Initialize RNNBlock
        :param args: arg.blocks.rnn?.yaml (see hydra configs).
        """
        self.rnn = nn.Sequential(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor
        :return: Output tensor
        """
        return self.rnn(x)

