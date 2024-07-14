from enum import Enum
from typing import Callable, Self
from functools import partial

import torch

from utils import TensorInstructions, TensorRef


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


    def __add__(self, other: Self | int | float):
        return self._single_inp_op(other, "__add__")
    
    def __iadd__(self, other: Self | int | float):
        return self._single_inp_op(other, "__iadd__")
    
    def __radd__(self, other: Self | int | float):
        return self._single_inp_op(other, "__radd__")
    
    def __sub__(self, other: Self | int | float):
        return self._single_inp_op(other, "__sub__")
    
    def __isub__(self, other: Self | int | float):
        return self._single_inp_op(other, "__isub__")
    
    def __rsub__(self, other: Self | int | float):
        return self._single_inp_op(other, "__rsub__")
    
    def __mul__(self, other: Self | int | float):
        return self._single_inp_op(other, "__mul__")
    
    def __imul__(self, other: Self | int | float):
        return self._single_inp_op(other, "__imul__")
    
    def __rmul__(self, other: Self | int | float):
        return self._single_inp_op(other, "__rmul__")

    def __truediv__(self, other: Self | int | float):
        return self._single_inp_op(other, "__truediv__")
    
    def __idiv__(self, other: Self | int | float):
        return self._single_inp_op(other, "__idiv__")
    
    def __rtruediv__(self, other: Self | int | float):
        return self._single_inp_op(other, "__rtruediv__")
    
    # Not supported yet
    def __floordiv__(self, other: Self | int | float):
        raise NotImplementedError("Floordiv is not supported yet!")
    
    def __ifloordiv__(self, other: Self | int | float):
        raise NotImplementedError("Floordiv is not supported yet!")

    def __rfloordiv__(self, other: Self | int | float):
        raise NotImplementedError("Floordiv is not supported yet!")

    def __mod__(self, other: Self | int | float):
        return self._single_inp_op(other, "__mod__")
    
    def __imod__(self, other: Self | int | float):
        return self._single_inp_op(other, "__imod__")
    
    def __rmod__(self, other: Self | int | float):
        return self._single_inp_op(other, "__rmod__")
    
    def __pow__(self, other: Self | int | float):
        return self._single_inp_op(other, "__pow__")
    
    def __ipow__(self, other: Self | int | float):
        return self._single_inp_op(other, "__ipow__")
    
    def __rpow__(self, other: Self | int | float):
        return self._single_inp_op(other, "__rpow__")
    
    def __rshift__(self, other: Self | int | float):
        return self._single_inp_op(other, "__rshift__")
    
    def __irshift__(self, other: Self | int | float):
        return self._single_inp_op(other, "__irshift__")

    def __lshift__(self, other: Self | int | float):
        return self._single_inp_op(other, "__lshift__")
    
    def __ilshift__(self, other: Self | int | float):
        return self._single_inp_op(other, "__ilshift__")
    
    def __and__(self, other: Self | int | float):
        return self._single_inp_op(other, "__and__")
    
    def __iand__(self, other: Self | int | float):
        return self._single_inp_op(other, "__iand__")
    
    def __or__(self, other: Self | int | float):
        return self._single_inp_op(other, "__or__")
    
    def __ior__(self, other: Self | int | float):
        return self._single_inp_op(other, "__ior__")
    
    def __xor__(self, other: Self | int | float):
        return self._single_inp_op(other, "__xor__")
    
    def __ixor__(self, other: Self | int | float):
        return self._single_inp_op(other, "__ixor__")
    
    def __lt__(self, other: Self | int | float):
        return self._single_inp_op(other, "__lt__")
    
    def __le__(self, other: Self | int | float):
        return self._single_inp_op(other, "__le__")
    
    def __gt__(self, other: Self | int | float):
        return self._single_inp_op(other, "__gt__")
    
    def __ge__(self, other: Self | int | float):
        return self._single_inp_op(other, "__ge__")
    
    def __eq__(self, other: Self | int | float):
        return self._single_inp_op(other, "__eq__")
    
    def __ne__(self, other: Self | int | float):
        return self._single_inp_op(other, "__ne__")
    

    def _single_inp_op(self, other: Self | int | float, op_name: str):
        self_ref = TensorRef(id(self))
        other_ref = TensorRef(id(self)) if isinstance(other, STensor) else other
        other = other.tensor if isinstance(other, STensor) else other

        save_callback = self.callback(op_name, self_ref, (other_ref,))

        result = getattr(self.tensor, op_name)(other)
        result = STensor(result, self.callback)
        save_callback(id(result))

        return result
