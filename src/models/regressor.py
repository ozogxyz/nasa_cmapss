from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from .blocks import Conv1DBlock, LSTMBlock


class ConvLSTMNet(pl.LightningModule):
    """
    A hybrid convolutional and recurrent neural network for remining useful life (RUL) regression.
    """

    def __init__(self, args: Any):
        super(ConvLSTMNet, self).__init__()
        self.conv1 = Conv1DBlock(args['conv_block_1'])

        self.conv2 = Conv1DBlock(args['conv_block_2'])

        self.lstm = LSTMBlock(args['lstm'])

        self.fc = nn.Linear(64, 1)
        self.loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param inputs: Input tensor.
        :return: Output tensor.
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.lstm(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        One training step.

        :param batch:
        :param batch_idx:
        :return:
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        One evaluation step.

        :param batch:
        :param batch_idx:
        :return:
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        One evaluation step.

        :param batch:
        :param batch_idx:
        :return:
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        :return:
        """
        return self.optimizer
