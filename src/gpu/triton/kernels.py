import triton
import triton.language as tl #type:ignore

@triton.jit
def add_kernel(
    a_ptr, b_ptr, o_ptr, n, block_size
):
    pid = tl.program_id(axis=0)

    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    mask = offsets < n

    x_d = tl.load(a_ptr + offsets, mask=mask)
    y_d = tl.load(b_ptr + offsets, mask=mask)

    output = x_d + y_d
    tl.store(o_ptr + offsets, output, mask=mask)

@triton.jit
def sub_kernel(
    a_ptr, b_ptr, o_ptr, n, block_size
):
    ## Apparently axis=0 means its a 1D launch grid
    ## tl.program_id just grabs a program_id? Idk what it does
    pid = tl.program_id(axis=0)

    ## Get the start of the block, ig its equivalent to blockIdx
    ## I'd say this is establishing boundaries 
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n

    ## Load is equivalent to malloc + memcpy, its interesting how triton just does everything 
    ## At one place, kinda convinient. 
    x_d = tl.load(a_ptr + offsets, mask=mask)
    y_d = tl.load(b_ptr + offsets, mask=mask)

    ## Just like cuda, we just do one line for the actual operation. 
    ## But the boundary checks are already handled during mem alloc
    ## So ig thats nicer?
    output = x_d - y_d
    
    ## Again, one liner equivalent to so much CUDA jargon, i'm drooling
    tl.store(o_ptr + offsets, output, mask=mask)

