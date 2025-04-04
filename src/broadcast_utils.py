import numpy as np

## Local imports
from .pico import Tensor

## Broadcasting utils - have to implement this on my own
def check_shapes(a: Tensor, b: Tensor):
    return a.shape == b.shape

def broadcast(a: Tensor, b: Tensor):
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        a_b, b_b = np.broadcast_arrays(a, b)
        return Tensor( data=a_b ), Tensor( data=b_b )
