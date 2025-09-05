import torch
import triton
from .kernels import add_kernel, sub_kernel

DEVICE = triton.runtime.driver.active.get_active_torch_device()

## Triton section
def _triton_add(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and out.device == DEVICE
    n_elements = out.numel()

    grid = grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)

    return out

def _triton_sub(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and out.device == DEVICE
    n_elements = out.numel()

    grid = grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    sub_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)

    return out