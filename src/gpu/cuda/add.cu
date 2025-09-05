#include <cuda_runtime.h>
#include <iostream>

__global__ void add_kernel(
    float* M, 
    float* N, 
    float* O, 
    int N
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        O[idx] = M[idx] + N[idx];
    }
}

__global__  sub_kernel(
    float* M, 
    float* N, 
    float* O, 
    int N
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        O[idx] = M[idx] - N[idx];
    }
}