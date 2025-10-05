# uhop/examples/matmul_benchmark.py
import time
import numpy as np
import matplotlib.pyplot as plt
from uhop import Tensor, Executor
from uhop.core.monitor import monitor

def run_matmul_benchmark():
    sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    results = {
        'cpu': {'time': [], 'flops': []},
        'cuda': {'time': [], 'flops': []}
    }
    
    print("="*50)
    print("Matrix Multiplication Benchmark")
    print("="*50)
    
    for M, N in sizes:
        print(f"\nRunning {M}x{M} matrix multiplication...")
        a_np = np.random.randn(M, N).astype(np.float32)
        b_np = np.random.randn(N, M).astype(np.float32)
        
        # CPU Execution
        cpu_exec = Executor('cpu')
        a_cpu, b_cpu = Tensor(a_np), Tensor(b_np)
        start = time.perf_counter()
        c_cpu = cpu_exec.matmul(a_cpu, b_cpu)
        cpu_time = time.perf_counter() - start
        flops = 2 * M * M * N  # FLOP count
        
        results['cpu']['time'].append(cpu_time)
        results['cpu']['flops'].append(flops / cpu_time / 1e9)  # GFLOP/s
        print(f"CPU: {cpu_time:.4f}s | {flops/cpu_time/1e9:.2f} GFLOP/s")
        
        # GPU Execution
        gpu_exec = Executor('cuda')
        a_gpu, b_gpu = a_cpu.gpu(), b_cpu.gpu()
        start = time.perf_counter()
        c_gpu = gpu_exec.matmul(a_gpu, b_gpu)
        cuda_time = time.perf_counter() - start
        
        results['cuda']['time'].append(cuda_time)
        results['cuda']['flops'].append(flops / cuda_time / 1e9)  # GFLOP/s
        print(f"CUDA: {cuda_time:.4f}s | {flops/cuda_time/1e9:.2f} GFLOP/s")
        
        # Validate results
        diff = np.max(np.abs(c_cpu.data - c_gpu.cpu().data))
        print(f"Validation: Max difference = {diff:.6f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Execution Time
    plt.subplot(1, 2, 1)
    plt.plot([f"{s[0]}x{s[1]}" for s in sizes], results['cpu']['time'], 'o-', label='CPU')
    plt.plot([f"{s[0]}x{s[1]}" for s in sizes], results['cuda']['time'], 'o-', label='CUDA')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (s)')
    plt.title('Execution Time')
    plt.legend()
    plt.grid(True)
    
    # Throughput
    plt.subplot(1, 2, 2)
    plt.plot([f"{s[0]}x{s[1]}" for s in sizes], results['cpu']['flops'], 'o-', label='CPU')
    plt.plot([f"{s[0]}x{s[1]}" for s in sizes], results['cuda']['flops'], 'o-', label='CUDA')
    plt.xlabel('Matrix Size')
    plt.ylabel('GFLOP/s')
    plt.title('Computational Throughput')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('matmul_benchmark.png')
    print("\nBenchmark results saved to matmul_benchmark.png")

if __name__ == "__main__":
    run_matmul_benchmark()