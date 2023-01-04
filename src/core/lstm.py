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


class LSTM(nn.Module):
    def __init__(
        self,
        input_length: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        """
        :param input_length: Input batch size.
        :param num_classes: Size of the desired output 
        :param hidden_size: Size of the hidden state in LSTM layers.
        :param num_layers: Determines how many LSTM layers will be stacked.
        """
        super().__init__()

        # LSTM Layers
        self.lstm = nn.LSTM(input_length, hidden_size, num_layers, dropout=dropout)
        self.tanh = nn.Tanh()

    def forward(self, input_sequence: Tensor):
        # Temporal Dependency Capture
        lstm_output, hidden_state = self.lstm(input_sequence)
        lstm_output = self.tanh(lstm_output)
        return lstm_output, hidden_state

