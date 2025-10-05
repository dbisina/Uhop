import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pycuda.driver as cuda_driver
cuda_driver.init()

# Add the parent directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from uhop import Tensor, Executor
from uhop.adapters import CUDA_AVAILABLE

def run_performance_comparison():
    """Run comprehensive performance comparison between CPU and GPU"""
    print("Performance Comparison: CPU vs GPU")
    print("=" * 50)
    
    # Test configurationsz
    matmul_sizes = [128, 256, 512, 1024]
    conv_batch_sizes = [1, 4, 8, 16]
    
    # Results storage
    results = {
        'matmul': {'cpu': [], 'gpu': []},
        'conv2d': {'cpu': [], 'gpu': []}
    }
    
    # Matrix Multiplication tests
    print("\nMatrix Multiplication Tests:")
    print("-" * 30)
    for size in matmul_sizes:
        print(f"Testing {size}x{size}...")
        
        # Create matrices
        a_np = np.random.randn(size, size).astype(np.float32)
        b_np = np.random.randn(size, size).astype(np.float32)
        
        a = Tensor(a_np)
        b = Tensor(b_np)
        
        # CPU benchmark
        cpu_exec = Executor('cpu')
        start = time.time()
        result_cpu = cpu_exec.matmul(a, b)
        cpu_time = time.time() - start
        results['matmul']['cpu'].append(cpu_time)
        
        # GPU benchmark
        if CUDA_AVAILABLE:
            gpu_exec = Executor('cuda')
            start = time.time()
            result_gpu = gpu_exec.matmul(a.gpu(), b.gpu())
            cuda_driver.Context.synchronize()
            gpu_time = time.time() - start
            results['matmul']['gpu'].append(gpu_time)
            print(f"  CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {cpu_time/gpu_time:.2f}x")
        else:
            results['matmul']['gpu'].append(None)
            print(f"  CPU: {cpu_time:.4f}s, GPU: Not available")
    
    # Convolution tests
    print("\n2D Convolution Tests:")
    print("-" * 30)
    channels, height, width = 3, 128, 128
    kernel_size, out_channels = 3, 16
    
    for batch_size in conv_batch_sizes:
        print(f"Testing batch size {batch_size}...")
        
        # Create input and kernel
        input_np = np.random.randn(batch_size, channels, height, width).astype(np.float32)
        kernel_np = np.random.randn(out_channels, channels, kernel_size, kernel_size).astype(np.float32)
        
        input_tensor = Tensor(input_np)
        kernel_tensor = Tensor(kernel_np)
        
        # CPU benchmark
        cpu_exec = Executor('cpu')
        start = time.time()
        result_cpu = cpu_exec.conv2d(input_tensor, kernel_tensor, stride=1, padding=1)
        cpu_time = time.time() - start
        results['conv2d']['cpu'].append(cpu_time)
        
        # GPU benchmark
        if CUDA_AVAILABLE:
            gpu_exec = Executor('cuda')
            start = time.time()
            result_gpu = gpu_exec.conv2d(input_tensor.gpu(), kernel_tensor.gpu(), stride=1, padding=1)
            cuda_driver.Context.synchronize()
            gpu_time = time.time() - start
            results['conv2d']['gpu'].append(gpu_time)
            print(f"  CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {cpu_time/gpu_time:.2f}x")
        else:
            results['conv2d']['gpu'].append(None)
            print(f"  CPU: {cpu_time:.4f}s, GPU: Not available")
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matrix multiplication plot
    ax1.plot(matmul_sizes, results['matmul']['cpu'], 'o-', label='CPU', linewidth=2)
    if CUDA_AVAILABLE:
        ax1.plot(matmul_sizes, results['matmul']['gpu'], 'o-', label='GPU', linewidth=2)
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Matrix Multiplication Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Convolution plot
    ax2.plot(conv_batch_sizes, results['conv2d']['cpu'], 'o-', label='CPU', linewidth=2)
    if CUDA_AVAILABLE:
        ax2.plot(conv_batch_sizes, results['conv2d']['gpu'], 'o-', label='GPU', linewidth=2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('2D Convolution Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as performance_comparison.png")
    
    return results

if __name__ == "__main__":
    results = run_performance_comparison()