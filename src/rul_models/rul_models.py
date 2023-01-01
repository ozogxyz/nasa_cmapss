from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import MeanSquaredError


class RulModel(pl.LightningModule):
    """
    A hybrid convolutional and recurrent neural network for remining useful life (RUL) regression.
    """

    def __init__(self, args: Any):
        super(RulModel, self).__init__()

        self.conv1 = args.blocks.conv1
        self.conv2 = args.blocks.conv2
        self.maxpool = args.blocks.maxpool
        self.lstm = args.blocks.lstm
        self.fc = args.blocks.regressor

        self.loss = MeanSquaredError(squared=False)
        self.optimizer = torch.optim.Adam(self.parameters(), args.hparams.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        :param x: input tensor
        :return: output tensor
        """
        print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.lstm(x)
        x = torch.flatten(x[-1], start_dim=1)
        x = self.fc(x)
        return x.flatten()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Forward pass through the model.

        :param batch: input batch
        :param batch_idx: batch index
        :return: output batch
        """
        x, y = batch
        y_hat = self(x)
        print(y_hat.size(), y.size())
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Forward pass through the model.

        :param batch: input batch
        :param batch_idx: batch index
        :return: output batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Forward pass through the model.

        :param batch: input batch
        :param batch_idx: batch index
        :return: output batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self) -> Any:
        """
        Configure the optimizer.

        :return: optimizer
        """
        return self.optimizer
