# uhop/examples/relu_benchmark.py
import time
import numpy as np
import matplotlib.pyplot as plt
from uhop import Tensor, Executor

def run_relu_benchmark():
    sizes = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    results = {
        'cpu': {'time': [], 'throughput': []},
        'cuda': {'time': [], 'throughput': []}
    }
    
    print("="*50)
    print("ReLU Activation Benchmark")
    print("="*50)
    
    for size in sizes:
        print(f"\nRunning ReLU on {size:,} elements...")
        data_np = np.random.randn(size).astype(np.float32)
        
        # CPU Execution
        cpu_exec = Executor('cpu')
        t_cpu = Tensor(data_np)
        start = time.perf_counter()
        result_cpu = cpu_exec.relu(t_cpu)
        cpu_time = time.perf_counter() - start
        throughput = size / cpu_time / 1e6  # Millions of elements per second
        
        results['cpu']['time'].append(cpu_time)
        results['cpu']['throughput'].append(throughput)
        print(f"CPU: {cpu_time:.4f}s | {throughput:.2f} M elements/s")
        
        # GPU Execution
        gpu_exec = Executor('cuda')
        t_gpu = t_cpu.gpu()
        start = time.perf_counter()
        result_gpu = gpu_exec.relu(t_gpu)
        cuda_time = time.perf_counter() - start
        throughput = size / cuda_time / 1e6  # Millions of elements per second
        
        results['cuda']['time'].append(cuda_time)
        results['cuda']['throughput'].append(throughput)
        print(f"CUDA: {cuda_time:.4f}s | {throughput:.2f} M elements/s")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Execution Time
    plt.subplot(1, 2, 1)
    plt.plot([f"{s//1_000_000}M" for s in sizes], results['cpu']['time'], 'o-', label='CPU')
    plt.plot([f"{s//1_000_000}M" for s in sizes], results['cuda']['time'], 'o-', label='CUDA')
    plt.xlabel('Tensor Size')
    plt.ylabel('Time (s)')
    plt.title('Execution Time')
    plt.legend()
    plt.grid(True)
    
    # Throughput
    plt.subplot(1, 2, 2)
    plt.plot([f"{s//1_000_000}M" for s in sizes], results['cpu']['throughput'], 'o-', label='CPU')
    plt.plot([f"{s//1_000_000}M" for s in sizes], results['cuda']['throughput'], 'o-', label='CUDA')
    plt.xlabel('Tensor Size')
    plt.ylabel('Throughput (M elements/s)')
    plt.title('Processing Throughput')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('relu_benchmark.png')
    print("\nBenchmark results saved to relu_benchmark.png")

if __name__ == "__main__":
    run_relu_benchmark()