from typing import Any, Dict, Optional

import torch
import torchmetrics
import torch.nn as nn
from pytorch_lightning import LightningModule

from blocks import Conv1DBlock, LSTMBlock


class Cnn1dLSTM(LightningModule):
    def __init__(self, args: Any):
        super(Cnn1dLSTM, self).__init__()
        self.args = args
        self.lstm = LSTM(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
            batch_first=True,
        )
        self.conv1d = Conv1D(
            in_channels=args.input_size,
            out_channels=args.hidden_size,
            kernel_size=1,
            padding=0,
            bias=True,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
            batch_first=True,
            num_layers=args.num_layers,
            nonlinearity="relu",
            weight_norm=args.weight_norm,
            bias_norm=args.bias_norm,
        )

        self.linear = nn.Linear(args.hidden_size, args.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            verbose=True,
        )

        self.model_to_device()
        self.train()
        self.to(args.device)

        # Max-Pooling Layer
        if args.pooling == "max":
            self.pooling = torch.nn.MaxPool1d(
                kernel_size=2, stride=2, return_indices=True
            )
        elif args.pooling == "avg":
            self.pooling = torch.nn.AvgPool1d(kernel_size=2, stride=1)
        else:
            self.pooling = None
        
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        # Fully Connected Linear Layers for Regression
        self.fc1 = torch.nn.Linear(args.hidden_size, args.num_classes)        
        self.fc2 = torch.nn.Linear(args.hidden_size, 1)


    def model_to_device(self) -> None:
        """
        Move model parameters and optimizer state onto the correct device.
        """
        self.lstm.to(self.args.device)
        self.conv1d.to(self.args.device)
        self.linear.to(self.args.device)
        self.criterion.to(self.args.device)
        self.optimizer.to(self.args.device)
        self.scheduler.to(self.args.device)
        self.model.to(self.args.device)
    

    def forward(self, time_series):
        """
        Args:
            time_series (torch.Tensor): Input time series of shape (batch_size, num_features)

        Returns:
            torch.Tensor: Predicted class probabilities of shape (batch_size, num_classes)
        """
        # (batch_size, num_features) -> (batch_size, num_classes)
        logits = self.lstm(time_series)
        # (batch_size, num_classes) -> (batch_size, num_classes)
        logits = self.conv1d(logits)
        # (batch_size, num_classes) -> (batch_size, num_classes)
        logits = self.linear(logits)
        # (batch_size, num_classes) -> (batch_size, num_classes)
        if self.pooling is not None:
            logits = self.pooling(logits)
        # (batch_size, num_classes) -> (batch_size, num_classes)
        logits = self.fc1(logits)
        # (batch_size, num_classes) -> (batch_size, num_classes
        logits = self.fc2(logits)
        # (batch_size, num_classes) -> (batch_size, num_classes)
        return logits

    def training_step(self, batch, batch_idx):
        """
        One training step for the model

        Args:
            batch (torch.Tensor): Input batch of shape (batch_size, num_features)
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss scalar value
        """
        time_series = batch
        logits = self.forward(time_series)
        loss = self.criterion(logits, time_series)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        One validation step for the model

        Args:
            batch (torch.Tensor): Input batch of shape (batch_size, num_features)
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss scalar value
        """
        time_series = batch
        logits = self.forward(time_series)
        loss = self.criterion(logits, time_series)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        One test step for the model

        Args:
            batch (torch.Tensor): Input batch of shape (batch_size, num_features)
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss scalar value
        """
        time_series = batch
        logits = self.forward(time_series)
        loss = self.criterion(logits, time_series)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Set optimizer and learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size,
        gamma=self.args.gamma)
        return [optimizer], [scheduler]

    
