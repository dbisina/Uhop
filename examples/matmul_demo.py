# uhop/examples/matmul_demo.py
from uhop.core.tensor import Tensor
from uhop.core.executor import Executor
import numpy as np

# Generate random matrices
a_np = np.random.randn(256, 128).astype(np.float32)
b_np = np.random.randn(128, 64).astype(np.float32)

# Create Tensors
a = Tensor(a_np)
b = Tensor(b_np)

# CPU Execution
cpu_exec = Executor('cpu')
c_cpu = cpu_exec.matmul(a, b).cpu().data

# GPU Execution
gpu_exec = Executor('cuda')
c_gpu = gpu_exec.matmul(a, b).cpu().data

# Validation
diff = np.max(np.abs(c_cpu - c_gpu))
print(f"Max difference: {diff:.6f}")
print(f"CPU shape: {c_cpu.shape}, GPU shape: {c_gpu.shape}")
print(f"Results {'match' if diff < 1e-5 else 'differ'}")


// uhop/kernels/matmul_kernel.cu
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
} 
