from typing import Callable
from enum import Enum

import torch


class Strategy(Enum):
    # We are going shard and replicate only dim 0 for now!
    Shard = 0
    Replicate = 1



class STensor:
    # STensor works exactly like PyTorch Dtensor except one main difference,
    # for every operation it is going to it will send a message via callback
    # function to the distribution center.

    def __init__(self, tensor: torch.Tensor, strategy: Strategy, callback: Callable) -> None:
        self.tensor = tensor
        self.strategy = strategy
        self.callback = callback



    