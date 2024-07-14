import os
from enum import Enum
import multiprocessing as mp
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed._tensor import distribute_tensor, Shard, Replicate, DTensor, init_device_mesh, DeviceMesh

import utils
from utils import TensorInstructions
from tensor import STensor

class Instructions(Enum):
    SHARD = 0 # Shard tensor across devices
    REPLICATE = 1 # Replicate tensor across devices
    BROADCAST = 2 # Broadcast tensor across tensor
    RUN_OP = 3


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
        stensor = STensor(dtensor, self._tensor_callback)
        self.tensor_ref[id(stensor)] = self.tensor_counter+1
        self.tensor_counter += 2
        
        return stensor

    def replicateTensor(self, tensor: torch.Tensor):
        # Broadcast
        self._send_instrcs(Instructions.BROADCAST, list(tensor.shape))
        dist.broadcast(tensor, src = 0)
        self.tensor_ref[id(tensor)] = self.tensor_counter

        # Replicate
        self._send_instrcs(Instructions.REPLICATE, self.tensor_counter)
        dtensor = distribute_tensor(tensor, self.device_mesh, [Replicate()])
        stensor = STensor(dtensor, self._tensor_callback)
        self.tensor_ref[id(stensor)] = self.tensor_counter+1
        self.tensor_counter += 2

        return stensor



    def _send_instrcs(self, instruction: Instructions, arg1: Any):
        dist.broadcast_object_list([instruction, None, arg1, None])

    
    def _tensor_callback(self, method_name:str, base_tensor: int, args: list[int]):
        base_tensor_id = self.tensor_ref[base_tensor]
        args = [self.tensor_ref[arg] for arg in args]
        dist.broadcast_object_list([Instructions.RUN_OP, method_name, base_tensor_id, args])
        
    
    def _run_method(self, method_name: str, tensor: DTensor, args: tuple[DTensor, ...]):
        res = getattr(tensor, method_name)(*args)


    def _initilize_distributed(self, rank:int, world_size:int):
        utils.init_distributed(rank=rank, world_size=self.nproc, device="cpu")
        self.device_mesh = init_device_mesh("cpu", (world_size,))
        

    def process(self, rank: int):
        # Initialize backend
        self._initilize_distributed(rank, self.nproc)

        while True:
            instruction = [None, None, None, None]
            dist.broadcast_object_list(instruction)
            base_instruction, instruction, base_tensor, args = instruction

            match base_instruction:
                case Instructions.BROADCAST:
                    assert isinstance(base_tensor, Sequence)
                    tensor = torch.empty(base_tensor)
                    dist.broadcast(tensor, src = 0)
                    self.tensor_ref[self.tensor_counter] = tensor
                    self.tensor_counter += 1

                case Instructions.SHARD:
                    tensor = self.tensor_ref[base_tensor]
                    dtensor = distribute_tensor(tensor, self.device_mesh, [Shard(0)])
                    self.tensor_ref[self.tensor_counter] = dtensor
                    self.tensor_counter += 1

                case Instructions.REPLICATE:
                    tensor = self.tensor_ref[base_tensor]
                    dtensor = distribute_tensor(tensor, self.device_mesh, [Replicate()])
                    self.tensor_ref[self.tensor_counter] = dtensor
                    self.tensor_counter += 1

                case Instructions.RUN_OP:
                    tensor = self.tensor_ref[base_tensor]
                    args = [self.tensor_ref[idx] for idx in args]
                    self._run_method(instruction, tensor, args)

                case _:
                    raise NotImplementedError()