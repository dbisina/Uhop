import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

# Simple CUDA test
mod = SourceModule("""
    __global__ void add(float *a, float *b, float *c, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
""")

add_func = mod.get_function("add")

# Test data
n = 100
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

# Allocate GPU memory
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy data to GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Launch kernel
block_size = 256
grid_size = (n + block_size - 1) // block_size
add_func(a_gpu, b_gpu, c_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

# Copy result back
cuda.memcpy_dtoh(c, c_gpu)

# Verify result
expected = a + b
if np.allclose(c, expected):
    print("✓ CUDA test passed!")
else:
    print("✗ CUDA test failed!")

print("Test completed")