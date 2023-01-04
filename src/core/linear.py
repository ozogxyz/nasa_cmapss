from typing import Optional

from torch import Tensor
import torch.nn as nn

# PYTORCH_ENABLE_MPS_FALLBACK = 1

# # Check first that CUDA then MPS is available, if not fallback to CPU
# if torch.cuda.is_available():
#     device = "cuda:0"
# elif not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")
# else:
#     device = torch.device("mps")


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Linear Layer
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input: Tensor):
        return self.linear(input)
    