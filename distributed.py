import os
from enum import Enum
import multiprocessing as mp
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed._tensor import distribute_tensor, Shard, Replicate, DTensor, init_device_mesh, DeviceMesh

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
        self.tensor_ref: dict[int, int] = {}

        ctx = mp.get_context("spawn")

        # As a start we are going to utilize Queue, but we need to efficiently broadcast messages
        processes = [ctx.Process(target = self.process, args=(i+1,)) for i in range(3)]
        for process in processes:
            process.start()
        
        self._initilize_distributed(rank=0, world_size=self.nproc)
        self.processes = processes


    def shardTensor(self, tensor: torch.Tensor):
        # Broadcast
        self._send_instrcs(Instructions.BROADCAST, list(tensor.shape))
        dist.broadcast(tensor, src = 0)
        self.tensor_ref[id(tensor)] = self.tensor_counter

        # Shrad
        self._send_instrcs(Instructions.SHARD, self.tensor_counter)
        dtensor = distribute_tensor(tensor, self.device_mesh, [Shard(0)])
        self.tensor_ref[id(dtensor)] = self.tensor_counter+1
        self.tensor_counter += 2
        return dtensor

    def replicateTensor(self, tensor: torch.Tensor):
        # Broadcast
        self._send_instrcs(Instructions.BROADCAST, list(tensor.shape))
        dist.broadcast(tensor, src = 0)
        self.tensor_ref[id(tensor), self.tensor_counter]

        # Replicate
        self._send_instrcs(Instructions.REPLICATE, self.tensor_counter, [Replicate()])
        dtensor = distribute_tensor(tensor, mesh = self.device_mesh)
        self.tensor_ref[id(dtensor), self.tensor_counter+1]
        self.tensor_counter += 2
        return dtensor



    def _send_instrcs(self, instruction: Instructions, arg1: Any):
        dist.broadcast_object_list((instruction, arg1))


    def _initilize_distributed(self, rank:int, world_size:int):
        utils.init_distributed(rank=rank, world_size=self.nproc, device="cpu")
        self.device_mesh = init_device_mesh("cpu", (world_size,))
        

    def process(self, rank: int):
        # Initialize backend
        self._initilize_distributed(rank, self.nproc)

        while True:
            instruction = [None, None]
            dist.broadcast_object_list(instruction)
            instruction, arg = instruction

            match instruction:
                case Instructions.BROADCAST:
                    assert isinstance(arg, Sequence)
                    tensor = torch.empty(arg)
                    dist.broadcast(tensor, src = 0)
                    self.tensor_ref[self.tensor_counter] = tensor
                    self.tensor_counter += 1

                case Instructions.SHARD:
                    tensor = self.tensor_ref[arg]
                    dtensor = distribute_tensor(tensor, self.device_mesh, [Shard(0)])
                    self.tensor_ref[self.tensor_counter] = dtensor
                    self.tensor_counter += 1

                case Instructions.REPLICATE:
                    tensor = self.tensor_ref[arg]
                    dtensor = distribute_tensor(tensor, self.device_mesh, [Replicate()])
                    self.tensor_ref[self.tensor_counter] = dtensor
                    self.tensor_counter += 1
