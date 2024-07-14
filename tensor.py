from typing import Callable, Self
from enum import Enum

import torch

from utils import TensorInstructions


class Strategy(Enum):
    # We are going shard and replicate only dim 0 for now!
    Shard = 0
    Replicate = 1



class STensor:
    # STensor works exactly like PyTorch Dtensor except one main difference,
    # for every operation it is going to it will send a message via callback
    # function to the distribution center.

    def __init__(self, tensor: torch.Tensor, callback: Callable) -> None:
        self.tensor = tensor
        self.callback = callback



    def __add__(self, other: Self) -> Self:
        self.callback(self.__add__.__name__, id(self), (id(other),))
        return self.tensor + other.tensor