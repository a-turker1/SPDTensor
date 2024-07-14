import os
from enum import Enum
from dataclasses import dataclass

import torch
import torch.distributed as dist

AVAILABLE_DEVICE_TYPES = ["cpu", "cuda"]


def init_distributed(rank: int, world_size: int, device: str = "cpu"):
    # Init torch distributed with suitable backend
    os.environ['RANK'] = str(rank)
    os.environ['MASTER_PORT'] = '6125'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['WORLD_SIZE'] = str(world_size)

    assert device in AVAILABLE_DEVICE_TYPES, f"Provided device type {device} is not available. Available device types: {AVAILABLE_DEVICE_TYPES}"

    backend_type = "gloo" if device == "cpu" else "nccl"
    if backend_type == "nccl":
        torch.cuda.set_device(rank)

    dist.init_process_group(backend=backend_type, rank=rank, world_size=world_size)



class TensorInstructions(Enum):
    ADD = 0

@dataclass
class TensorRef:
    id: int
    