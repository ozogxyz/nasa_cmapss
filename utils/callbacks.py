import torch
from typing import Optional
from pytorch_lightning import Callback


class PrintCallback(Callback):
  # CHECKPOINT_JOIN_CHAR = "-" 
  # CHECKPOINT_NAME_LAST = "last"
  # CHECKPOINT_STATE_BEST_SCORE = "checkpoint_callback_best_model_score"
  # CHECKPOINT_STATE_BEST_PATH = "checkpoin_callback_best_model_path"

  # def __init__(
  #   self,
  #   filepath: Optional[str] = None,
  #   monitor: Optional[str]= None,
  #   verbose: bool = True,
  #   save_last: Optional[bool] = None,
  #   save_top_k: Optional[int] = None,
  #   save_weights_only: bool = False,
  #   node: str = "auto", 
  #   period: int = 1,
  #   prefix: str = "",
  # ) -> None:
  #   super().__init__()
  
  def __init__(self) -> None:
    super().__init__()
    
  def on_train_start(self, trainer, pl_module):
    print("\n" + "<"*20 + "...Training is starting..." + ">"*20 + "\n")

  def on_train_end(self, trainer, pl_module):
    print("\n"+ "<"*20 + "...Training is ending..." + ">"*20 + "\n")