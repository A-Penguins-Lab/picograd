from .pico import Tensor
from .gpu.triton.dispatcher import add_kernel, sub_kernel
from .gpu.cuda.dispatcher import _triton_add, _triton_sub
from .modes import SET_GPU_BACKEND

gpu = 'cuda' if SET_GPU_BACKEND == 'cuda' else 'triton'

def is_available():
    if gpu == 'cuda':
        pass
    
    else:
        pass

def _pico_gpu_add(a: Tensor, b: Tensor):
    if gpu == 'cuda':
        pass
    
    else:
        _triton_add(a, b)

def _pico_gpu_sub(a: Tensor, b: Tensor):
    if gpu == 'cuda':
        pass
    
    else:
        _triton_sub()