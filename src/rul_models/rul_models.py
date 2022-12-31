from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from .blocks import Conv1DBlock, LSTMBlock


class RulModel(pl.LightningModule):
    """
    A hybrid convolutional and recurrent neural network for remining useful life (RUL) regression.
    """

    def __init__(self, args: Any):
        super(RulModel, self).__init__()

        self.conv1 = args.blocks.conv1
        self.conv2 = args.blocks.conv2
        self.lstm = args.blocks.lstm
        self.fc = args.blocks.regressor
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), args.hparams.lr)

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
