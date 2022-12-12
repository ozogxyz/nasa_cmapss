from typing import Any, Dict, Optional

import torch
import torchmetrics
import torch.nn as nn
from pytorch_lightning import LightningModule

PYTORCH_ENABLE_MPS_FALLBACK = 1

# Check first that CUDA then MPS is available, if not fallback to CPU
if torch.cuda.is_available():
    device = "cuda:0"
elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    device = torch.device("mps")


class Cnn1dLSTM(LightningModule):
    def __init__(
        self,
        batch_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        maxpool_kernel: int,
        num_classes:  int,
        hidden_size: int,
        num_layers: int,
        maxpool_stride: Optional[int],
        window_size: Optional[int],
        lr: Optional[int],
    ):
        """
        :param batch_size: Input batch size.
        :param in_channels: Input channels for Conv1d, number of features (14 in FD001).
        :param out_channels: Output channels for 1st Conv1d, total conv output will be 2 * out_channels.
        :param kernel_size: Kernel size for 1st Conv1d, will be kernel_size - 2 in the next later.
        :param maxpool_kernel: Kernel size for pooling layer.
        :param num_classes:
        :param hidden_size: Size of the hidden state in LSTM layers.
        :param num_layers: Determines how many LSTM layers will be stacked.
        :param maxpool_stride: Pooling stride.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.maxpool_kernel = maxpool_kernel
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.maxpool_stride = maxpool_stride
        self.batch_size = batch_size
        self.lr = lr
        
        self.metric = torchmetrics.MeanSquaredError(squared=False)
        self.save_hyperparameters()

        # Convolutinal Layers
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_2 = nn.Conv1d(out_channels, out_channels*2, kernel_size-2)
        conv_out_dim = window_size - kernel_size - 1
        
        # Max-Pooling Layer
        self.maxpool = nn.MaxPool1d(
            kernel_size=maxpool_kernel, stride=maxpool_stride)

        # LSTM Layers
        embedding_length = (conv_out_dim - maxpool_kernel) // maxpool_stride + 1
        self.lstm = nn.LSTM(embedding_length, hidden_size,
                            num_layers, dropout=0.2)

        # Fully Connected Linear Layers for Regression
        self.fc_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, time_series):
        """
        rul-datasets library gives batches of:
        n_batch x n_features(14) x window_size(30)

        :param time_series: tensor([batch_size, 14, 30])
        :return: predictions: tensor([batch_size])
        """

        # Feature Extraction
        features = torch.relu(self.conv_1(time_series))
        features = torch.relu(self.conv_2(features))

        # Max-Pooling
        features = self.maxpool(features)

        # Temporal Dependency Capture
        lstm_output, _ = self.lstm(features)
        lstm_output = torch.tanh(lstm_output)
        lstm_output = lstm_output[:, -1]

        # Regression
        output = self.fc_1(lstm_output)
        output = torch.relu(output)
        output = self.fc_2(output)
        return output.flatten()

    def training_step(self, batch):
        time_series, target = batch
        preds = self.forward(time_series)
        loss = self.metric(preds, target)
        self.log_dict({'RMSE': self.metric}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, target = batch
        preds = self.forward(features)
        loss = self.metric(preds, target)
        self.log_dict({"Validation loss": loss})

    def test_step(self, batch, batch_idx, dataloader_idx=1):
        features, target = batch
        preds = self.forward(features)
        loss = self.metric(preds, target)
        self.log_dict({'Test RMSE': loss}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return super().on_save_checkpoint(checkpoint)