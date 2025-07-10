# uhop/adapters/cuda_adapter.py
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import os

with open(os.path.join(os.path.dirname(__file__), '../kernels/matmul_kernel.cu'), 'r') as f:
    kernel_code = f.read()
mod = SourceModule(kernel_code)
matmul_kernel = mod.get_function("matmul_kernel")

from ..core.tensor import Tensor

def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matmul"
    M, K = a.shape
    _, N = b.shape

    c_gpu = gpuarray.empty((M, N), dtype=np.float32)

    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size

    matmul_kernel(
        a.data.gpudata, b.data.gpudata, c_gpu.gpudata,
        np.int32(M), np.int32(N), np.int32(K),
        block=(block_size, block_size, 1),
        grid=(grid_x, grid_y)
    )

    return Tensor(c_gpu, device='gpu')
