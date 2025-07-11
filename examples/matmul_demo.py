# uhop/examples/matmul_demo.py
from uhop import Tensor, Executor
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
