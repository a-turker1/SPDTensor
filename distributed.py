import os
from enum import Enum
import multiprocessing as mp
from typing import Any, Sequence

import torch
import torch.distributed as dist

import utils

class Instructions(Enum):
    SHARD = 0 # Shard tensor across devices
    REPLICATE = 1 # Replicate tensor across devices
    BROADCAST = 2 # Broadcast tensor across tensor


class DistributionCenter:
    # The Skynet. It handles communication between proceses and sends instructions.
    def __init__(self, nproc: int) -> None:
        self.nproc = nproc
        self.tensor_counter = 0 #??

        ctx = mp.get_context("spawn")

        # As a start we are going to utilize Queue, but we need to efficiently broadcast messages
        processes = [ctx.Process(target = self.process, args=(i+1,)) for i in range(3)]
        for process in processes:
            process.start()
        
        utils.init_distributed(rank=0, world_size=self.nproc, device="cpu")
        self.processes = processes


    def shardTensor(self, tensor: torch.Tensor):
        # Broadcast
        self._send_instrcs(Instructions.BROADCAST, list(tensor.shape))
        dist.broadcast(tensor, src = 0)

        # Shrad
        pass
    

    def _send_instrcs(self, instruction: Instructions, arg1: Any):
        dist.broadcast_object_list((instruction, arg1))
        pass
    



    def process(self, rank: int):
        # Initialize backend
        utils.init_distributed(rank=rank, world_size=self.nproc, device="cpu")



        while True:
            instruction = [None, None]
            dist.broadcast_object_list(instruction)
            instruction, arg = instruction

            match instruction:
                case Instructions.BROADCAST:
                    assert isinstance(arg, Sequence)
                    tensor = torch.empty(arg)
                    dist.broadcast(tensor, src = 0)
                    print(f"This is rank: {os.environ['RANK']} and my tensor is:", tensor)

                case Instructions.SHARD:
                    pass
                case Instructions.REPLICATE:
                    pass
            
