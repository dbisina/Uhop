# uhop/examples/conv2d_benchmark.py
import time
import numpy as np
import matplotlib.pyplot as plt
from uhop import Tensor, Executor

def run_conv2d_benchmark():
    configs = [
        (1, 3, 128, 128, 3),   # Batch, Channels, H, W, Kernel
        (1, 3, 256, 256, 3),
        (1, 3, 512, 512, 3),
        (4, 3, 512, 512, 3)
    ]
    
    results = {
        'cpu': {'time': [], 'gflops': []},
        'cuda': {'time': [], 'gflops': []}
    }
    
    print("="*50)
    print("2D Convolution Benchmark")
    print("="*50)
    
    for batch, in_channels, height, width, kernel_size in configs:
        print(f"\nRunning Conv2D: {batch}x{in_channels}x{height}x{width} "
              f"with {kernel_size}x{kernel_size} kernel")
        
        # Create input and kernel
        input_np = np.random.randn(batch, in_channels, height, width).astype(np.float32)
        kernel_np = np.random.randn(16, in_channels, kernel_size, kernel_size).astype(np.float32)
        
        # Calculate FLOPs: 2 * batch * out_channels * out_h * out_w * in_channels * kernel_size^2
        out_h = height - kernel_size + 1
        out_w = width - kernel_size + 1
        flops = 2 * batch * 16 * out_h * out_w * in_channels * kernel_size * kernel_size
        
        # CPU Execution
        cpu_exec = Executor('cpu')
        input_cpu = Tensor(input_np)
        kernel_cpu = Tensor(kernel_np)
        start = time.perf_counter()
        result_cpu = cpu_exec.conv2d(input_cpu, kernel_cpu)
        cpu_time = time.perf_counter() - start
        
        results['cpu']['time'].append(cpu_time)
        results['cpu']['gflops'].append(flops / cpu_time / 1e9)
        print(f"CPU: {cpu_time:.4f}s | {flops/cpu_time/1e9:.2f} GFLOP/s")
        
        # GPU Execution
        gpu_exec = Executor('cuda')
        input_gpu = input_cpu.gpu()
        kernel_gpu = kernel_cpu.gpu()
        start = time.perf_counter()
        result_gpu = gpu_exec.conv2d(input_gpu, kernel_gpu)
        cuda_time = time.perf_counter() - start
        
        results['cuda']['time'].append(cuda_time)
        results['cuda']['gflops'].append(flops / cuda_time / 1e9)
        print(f"CUDA: {cuda_time:.4f}s | {flops/cuda_time/1e9:.2f} GFLOP/s")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Execution Time
    plt.subplot(1, 2, 1)
    labels = [f"{c[2]}x{c[3]}" for c in configs]
    plt.plot(labels, results['cpu']['time'], 'o-', label='CPU')
    plt.plot(labels, results['cuda']['time'], 'o-', label='CUDA')
    plt.xlabel('Image Size')
    plt.ylabel('Time (s)')
    plt.title('Execution Time')
    plt.legend()
    plt.grid(True)
    
    # Computational Throughput
    plt.subplot(1, 2, 2)
    plt.plot(labels, results['cpu']['gflops'], 'o-', label='CPU')
    plt.plot(labels, results['cuda']['gflops'], 'o-', label='CUDA')
    plt.xlabel('Image Size')
    plt.ylabel('GFLOP/s')
    plt.title('Computational Throughput')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('conv2d_benchmark.png')
    print("\nBenchmark results saved to conv2d_benchmark.png")

if __name__ == "__main__":
    run_conv2d_benchmark()