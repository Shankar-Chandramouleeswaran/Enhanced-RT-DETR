from pprint import pprint
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from typing import Callable

__all__ = ['BaseConfig']

class BaseConfig(object):
    def __init__(self) -> None:
        super().__init__()
        self.task: str = None 
        self._model: nn.Module = None 
        self._criterion: nn.Module = None 
        self._optimizer: Optimizer = None 
        self._lr_scheduler: LRScheduler = None 
        self._train_dataloader: DataLoader = None 
        self._val_dataloader: DataLoader = None 
        self.epoches: int = 100  # Adjusted for more training epochs
        self.use_amp: bool = True  # Enable mixed precision training
        self.multi_scale_training: bool = True  # Added for multi-scale support
        self.data_augmentation: bool = True  # Enable data augmentation

    @property
    def model(self) -> nn.Module:
        return self._model 
    
    @model.setter
    def model(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module'
        self._model = m 

    @property
    def train_dataloader(self) -> DataLoader:
        if self.data_augmentation:
            print('Data augmentation enabled')
        return self._train_dataloader



