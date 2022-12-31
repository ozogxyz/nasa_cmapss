from typing import Any

import torch.nn as nn


class Conv1DBlock(nn.Module):
    """
    1D convolutional block
    """

    def __init__(self, args: Any):
        super(Conv1DBlock, self).__init__()
        """
        :param in_channels: Input channels for Conv1d, number of features (14 in FD001).
        :param out_channels: Output channels for 1st Conv1d, total conv output will be 2 * out_channels.
        :param kernel_size: Kernel size for 1st Conv1d, will be kernel_size - 2 in the next layer.
        """
        self.args = args
        self.conv1d = nn.Conv1d(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            dilation=args.dilation,
        )

        if args.batch_norm:
            self.bn1d = nn.BatchNorm1d(args.out_channels)

        if args.dropout is not None:
            self.dropout = nn.Dropout(args.dropout)
        else:
            self.dropout = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Input tensor
        :return: Output tensor
        """
        x = self.conv1d(x)
        if self.args.batch_norm:
            x = self.bn1d(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.relu(x)
        return


class LSTMBlock(nn.Module):
    """
    Long short-term memory (LSTM) block
    """

    def __init__(self, args: Any):
        super(LSTMBlock, self).__init__()
        """
        :param input_length: Input batch size.
        :param num_classes: Size of the desired output 
        :param hidden_size: Size of the hidden state in LSTM layers.
        :param num_layers: Determines how many LSTM layers will be stacked.
        """
        self.num_directions = 2 if args.num_layers > 1 else 1
        self.lstm = nn.LSTM(input_size=args.input_size
                            + args.hidden_size * args.num_directions,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            batch_first=args.batch_first,
                            bidirectional=args.bidirectional)

        if args.dropout is not None:
            self.dropout = nn.Dropout(args.dropout)
        else:
            self.dropout = None

        self.tanh = nn.Tanh()
        
    def forward(self, input_data):
        """
        :param input_data: Input data.
        :return: The output of LSTM.
        """
        output, _ = self.lstm(input_data)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.tanh(output)
        return output

