from typing import Optional
import torch
import torch.nn as nn

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


class Conv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 2
    ):
        """
        :param in_channels: Input channels for Conv1d, number of features (14 in FD001).
        :param out_channels: Output channels for 1st Conv1d, total conv output will be 2 * out_channels.
        :param kernel_size: Kernel size for 1st Conv1d, will be kernel_size - 2 in the next later.
        """
        super().__init__()

        # Convolutinal Layers
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, time_series):
        """
        rul-datasets library gives batches of:
        n_batch x n_features(14) x window_size(30)

        :param time_series: tensor([batch_size, 14, 30])
        :return: predictions: tensor([batch_size])
        """

        # Feature Extraction
        features = self.conv_1(time_series)
        return features
