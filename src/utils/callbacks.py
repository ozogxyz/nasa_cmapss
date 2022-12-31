import time
import torch
from typing import Any, Optional
from pytorch_lightning import Callback


class PrintCallback(Callback):
  """
  A callback that prints the metrics to stdout.
  """

  def __init__(self,  args: Any):
    super().__init__()
    self.args = args
    self.metrics = {}
    self.start_time = time.time()
    self.last_print_time = time.time()
    self.print_interval = self.args.print_interval

  def on_train_start(self, trainer, pl_module):
    self.metrics = {}
    self.last_print_time = time.time()

  def on_train_batch_end(self, trainer, pl_module, outputs,
                         batch, batch_idx, dataloader_idx):

    if batch_idx % self.print_interval == 0:
      metrics = trainer.callback_metrics
      for k, v in metrics.items():
        self.metrics[k] = v
        if self.last_print_time + self.args.print_interval < time.time.sleep(self.args.print_interval):
          self.last_print_time = time.time()
          self.print_metrics()
        self.metrics[k] = 0
      self.print_metrics()
    else:
      self.metrics = {}

  def print_metrics(self, name, value):
    if name not in self.metrics:
      self.metrics[name] = []
    self.metrics[name].append(value)
    if len(self.metrics[name]) == self.args.print_interval:
      self.metrics[name] = self.metrics[name][1:]
      self.metrics[name] = self.metrics[name][:self.args.print_interval]
      self.metrics[name] = self.metrics[name][:-1]
      if len(self.metrics[name]) > 0:
        self.metrics[name] = ''.join(self.metrics[name])
        print(f'{name}: {self.metrics[name]}')
        self.metrics[name] = []
        self.metrics[name] = self.metrics[name][1:]
        self.metrics[name] = self.metrics[name][:self.args.print_]
      else:
        print(f'{name}: {self.metrics[name]}')
        self.metrics[name] = []
        self.metrics[name] = self.metrics[name][1:]
