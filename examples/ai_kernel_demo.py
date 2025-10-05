# uhop/examples/ai_kernel_demo.py
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from uhop.core.ai_kernel_generator import AIKernelGenerator
    from uhop import Tensor, Executor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def run_ai_kernel_demo():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    # Configuration
    matrix_size = 1024
    input_shape = (matrix_size, matrix_size)
    output_shape = (matrix_size, matrix_size)
    
    print("="*50)
    print("AI Kernel Optimization Demo")
    print("="*50)
    
    # Create test data
    a_np = np.random.randn(*input_shape).astype(np.float32)
    b_np = np.random.randn(*input_shape).astype(np.float32)
    a = Tensor(a_np)
    b = Tensor(b_np)
    
    # Baseline CUDA performance
    gpu_exec = Executor('cuda')
    start = time.perf_counter()
    c_baseline = gpu_exec.matmul(a.gpu(), b.gpu())
    baseline_time = time.perf_counter() - start
    flops = 2 * matrix_size**3
    
    print(f"\nBaseline CUDA MatMul: {baseline_time:.6f}s")
    print(f"  Performance: {flops / baseline_time / 1e9:.2f} GFLOP/s")
    
    # AI Kernel Generation
    ai_generator = AIKernelGenerator(api_key="your-api-key-here", backend='cuda')
    
    constraints = {
        "operation": "matmul",
        "target_device": "NVIDIA GPU",
        "tile_sizes": [16, 32, 64],
        "max_registers": 64,
        "use_shared_memory": True
    }
    
    print("\nGenerating AI-optimized kernel...")
    kernel_code = ai_generator.generate_kernel("matmul", constraints)
    
    if kernel_code:
        print(f"Generated kernel size: {len(kernel_code)} characters")
        
        # Evaluate AI kernel
        print("\nEvaluating AI-optimized kernel...")
        exec_time, kernel_path, error = ai_generator.evaluate_kernel(
            kernel_code, "matmul", input_shape, output_shape
        )
        
        if exec_time < float('inf'):
            print(f"AI-optimized MatMul: {exec_time:.6f}s")
            print(f"  Performance: {flops / exec_time / 1e9:.2f} GFLOP/s")
            print(f"  Speedup: {baseline_time/exec_time:.2f}x")
            print(f"  Validation error: {error:.6f}")
            
            # Visualization
            labels = ['Baseline CUDA', 'AI-Optimized']
            times = [baseline_time, exec_time]
            perfs = [flops / baseline_time / 1e9, flops / exec_time / 1e9]
            
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.bar(labels, times, color=['blue', 'green'])
            plt.ylabel('Time (s)')
            plt.title('Execution Time')
            
            plt.subplot(1, 2, 2)
            plt.bar(labels, perfs, color=['blue', 'green'])
            plt.ylabel('GFLOP/s')
            plt.title('Computational Throughput')
            
            plt.tight_layout()
            plt.savefig('ai_kernel_optimization.png')
            print("\nResults saved to ai_kernel_optimization.png")
        else:
            print("Failed to evaluate AI-generated kernel")
    else:
        print("Failed to generate AI-optimized kernel")

if __name__ == "__main__":
    run_ai_kernel_demo()