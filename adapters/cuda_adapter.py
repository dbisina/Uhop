# uhop/adapters/cuda_adapter.py
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import os
from uhop.core.tensor import Tensor
from uhop.core.tuner import kernel_tuner

with open(os.path.join(os.path.dirname(__file__), '../kernels/matmul_kernel.cu'), 'r') as f:
    kernel_code = f.read()
mod = SourceModule(kernel_code)
matmul_kernel = mod.get_function("matmul_kernel")

with open(os.path.join(os.path.dirname(__file__), '../kernels/relu_kernel.cu'), 'r') as f:
    relu_kernel_code = f.read()
relu_mod = SourceModule(relu_kernel_code)
relu_kernel = relu_mod.get_function("relu_kernel")


def matmul(a, b, block_size=None):
    if block_size is None:
        block_size = kernel_tuner.get_optimal_block_size(
            'matmul', a.shape, b.shape
        )
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matmul"
    M, K = a.shape
    _, N = b.shape

    c_gpu = gpuarray.empty((M, N), dtype=np.float32)

    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size

    matmul_kernel(
        a.data, b.data, c_gpu,
        np.int32(M), np.int32(N), np.int32(K),
        block=(block_size, block_size, 1),
        grid=(grid_x, grid_y)
    )

    cuda.Context.synchronize()
    return Tensor(c_gpu, device='gpu')


def relu(a, block_size=256):
    size = a.data.size
    output = gpuarray.empty_like(a.data)
    
    grid_size = (size + block_size - 1) // block_size
    relu_kernel(
        a.data, output, np.int32(size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )
    return Tensor(output, device='gpu')


def conv2d(input, kernel, stride=1, padding=0, block_size=(8, 8, 4)):
    assert len(input.shape) == 4 and len(kernel.shape) == 4
    batch, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape
    assert kernel_h == kernel_w, "Only square kernels supported"
    
    # Calculate output dimensions
    out_h = (in_h - kernel_h + 2 * padding) // stride + 1
    out_w = (in_w - kernel_w + 2 * padding) // stride + 1
    
    # Create output buffer
    output_gpu = gpuarray.empty((batch, out_channels, out_h, out_w), dtype=np.float32)
    
    # Configure grid and block
    grid_x = (out_w + block_size[0] - 1) // block_size[0]
    grid_y = (out_h + block_size[1] - 1) // block_size[1]
    grid_z = (out_channels + block_size[2] - 1) // block_size[2]
    
    conv2d_kernel(
        input.data, kernel.data, output_gpu,
        np.int32(in_channels), np.int32(in_h), np.int32(in_w),
        np.int32(out_channels), np.int32(kernel_h),
        np.int32(out_h), np.int32(out_w),
        np.int32(stride), np.int32(padding),
        block=block_size,
        grid=(grid_x, grid_y, grid_z)
    )
    
    return Tensor(output_gpu, device='gpu')