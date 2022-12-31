from ast import Dict, List
from dataclasses import MISSING, dataclass
import os
from pathlib import Path, PurePath
import sys
from typing import Any

@dataclass
class DatasetConfig: 
    fd: int = 1
    path: str = PurePath(os.environ.get('HOME')) / ".rul_datasets/"
    name: str = f"{path.joinpath(path, 'CMAPSS')}"
    batch_size: int = 32
    window_size: int = 30

@dataclass
class HParamsConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    lr_scheduler: str = "cosine"
    gradient_clip_val: float = 0.5
    gradient_clip_algorithm: str = "value"
    patience: int = 10
    min_delta: float = 0.1
    log_dir: str = "./logs"
    name: str = "training"
    save_path: str = "./logs/training.log"
    resume: bool = True
    version: str = "0.1"
    git_sha: str = "{git rev-parse HEAD | cut -c 1-8}"
    git_branch: str = "main"

@dataclass 
class Dataset:
    config: DatasetConfig
    data: Dict[str, Any]


class ConvConfig:
    conv : List[Any] = MISSING
    # in_channels: int = 32
    # out_channels: int = 32
    # kernel_size: int = 3
    # stride: int = 1
    # padding: int = 1
    # bias: bool = False
    # groups: int = 1
    # activation: str = "relu"
    # batch_norm: bool = False
    # dropout: float = 0.0

@dataclass
class LSTMConfig:
    lstm: List[Any] = MISSING
    # input_size: int = 64
    # output_length: int = 64
    # hidden_size: int = 50
    # dropout: float = 0.25
    # batch_first: bool = True
    # bidirectional: bool = False
    # activation: str = "ReLU"
    # num_classes: int = 1
    # num_directions: int = 1
    # num_layers: int = 1

@dataclass
class BaseBlockConfig:
    conv1: ConvBlockConfig = ConvBlockConfig()
    conv2: ConvBlockConfig = ConvBlockConfig()
    lstm: LSTMConfig = LSTMConfig()

@dataclass
class BaseConfig:
    dataset: DatasetConfig = DatasetConfig()
    blocks: BaseBlockConfig = BaseBlockConfig()

