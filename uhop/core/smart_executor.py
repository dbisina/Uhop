# uhop/core/smart_executor.py
import os
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from uhop.core.tensor import Tensor
from uhop.adapters import cuda_adapter, cpu_adapter
from uhop.core.monitor import monitor, timed_operation
from uhop.core.ai_kernel_generator import AIKernelGenerator
from uhop.core.kernel_db import KernelDB

class SmartExecutor:
    def __init__(self, backend='cpu', api_key=None, use_ai=False):
        self.backend = backend
        self.use_ai = use_ai
        self.kernel_db = KernelDB()
        self.ai_kernel = AIKernelGenerator(api_key=api_key, backend=backend) if use_ai else None
        self.known_good = cpu_adapter

    @timed_operation
    def matmul(self, a, b):
        a_cpu, b_cpu = a.cpu(), b.cpu()
        input_shape = a.shape
        output_shape = (a.shape[0], b.shape[1])

        if self.backend == 'cpu':
            return self.known_good.matmul(a_cpu, b_cpu)

        elif self.backend == 'cuda':
            a_gpu, b_gpu = a.gpu(), b.gpu()

            # First try knowledge base
            best_kernel = self.kernel_db.get_best_kernel('matmul', input_shape, output_shape)
            if best_kernel:
                print(f"🚀 Using optimized kernel from knowledge base")
                return self._run_ai_kernel(
                    best_kernel['kernel_code'], 
                    'matmul', 
                    [a_gpu, b_gpu], 
                    input_shape, 
                    output_shape
                )

            # AI-generated kernel path
            if self.use_ai:
                constraints = {
                    "target_device": "NVIDIA GPU",
                    "tile_sizes": [8, 16, 32],
                    "operation": "matmul"
                }
                
                # Get or generate kernel
                kernel_code, db_entry = self.ai_kernel.get_or_generate_kernel(
                    'matmul', constraints, input_shape, output_shape
                )
                
                if kernel_code:
                    # Evaluate and refine
                    refined_kernel, exec_time, error = self.ai_kernel.evaluate_and_refine(
                        kernel_code, 'matmul', constraints, input_shape, output_shape
                    )
                    
                    if refined_kernel:
                        print(f"✅ AI Kernel validated | Time: {exec_time:.6f}s | Error: {error:.2e}")
                        return self._run_ai_kernel(
                            refined_kernel, 
                            'matmul', 
                            [a_gpu, b_gpu], 
                            input_shape, 
                            output_shape
                        )

            # Fallback to native CUDA
            return cuda_adapter.matmul(a_gpu, b_gpu)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _run_ai_kernel(self, kernel_code, operation, tensors, input_shape, output_shape):
        """Compile and execute an AI-generated kernel"""
        try:
            # Compile kernel
            mod = SourceModule(kernel_code)
            kernel_func = mod.get_function(f"{operation}_kernel")
            
            # Create output tensor
            if operation == "matmul":
                output = Tensor(np.zeros(output_shape), dtype=np.float32).gpu()
                tensors.append(output)
            
            # Get launch parameters
            block, grid = self._get_launch_params(operation, input_shape, output_shape)
            
            # Prepare arguments
            args = [t.data for t in tensors]
            if operation == "matmul":
                M, K = input_shape
                _, N = output_shape
                args.extend([np.int32(M), np.int32(N), np.int32(K)])
            
            # Execute kernel
            kernel_func(*args, block=block, grid=grid)
            cuda.Context.synchronize()
            
            # Track usage in knowledge base
            kernel_hash = hashlib.md5(kernel_code.encode()).hexdigest()
            self.kernel_db.increment_usage(f"{operation}_{kernel_hash}")
            
            return tensors[-1]  # Return output tensor
            
        except Exception as e:
            print(f"AI kernel execution failed: {str(e)}")
            # Fallback to native implementation
            if operation == "matmul":
                return cuda_adapter.matmul(tensors[0], tensors[1])
            return tensors[0]  # For unary ops

    def _get_launch_params(self, operation, input_shape, output_shape):
        """Determine launch configuration based on operation"""
        if operation == "matmul":
            M, _ = input_shape
            _, N = output_shape
            block = (16, 16, 1)
            grid_x = (N + block[0] - 1) // block[0]
            grid_y = (M + block[1] - 1) // block[1]
            return block, (grid_x, grid_y, 1)
        
        # Default configuration
        return (16, 16, 1), (1, 1, 1)
    
    def get_optimization_report(self):
        """Generate report of optimization benefits"""
        report = {
            "total_kernels": len(self.kernel_db.db),
            "top_optimized_ops": {}
        }
        
        for op in ["matmul", "relu", "conv2d"]:
            top_kernels = self.kernel_db.get_top_kernels(op)
            if top_kernels:
                report["top_optimized_ops"][op] = {
                    "best_time": top_kernels[0]["exec_time"],
                    "improvement": self._calculate_improvement(op, top_kernels[0])
                }
        
        return report
    
    def _calculate_improvement(self, operation, kernel_data):
        """Calculate speedup vs baseline"""
        # Create benchmark tensors
        input_shape = kernel_data["input_shape"]
        output_shape = kernel_data["output_shape"]
        
        if operation == "matmul":
            a = Tensor(np.random.randn(*input_shape))
            b = Tensor(np.random.randn(*input_shape))
            
            # Time baseline
            start = time.perf_counter()
            cuda_adapter.matmul(a.gpu(), b.gpu())
            cuda.Context.synchronize()
            baseline_time = time.perf_counter() - start
            
            return baseline_time / kernel_data["exec_time"]
        
        return 1.0  # Default no improvement