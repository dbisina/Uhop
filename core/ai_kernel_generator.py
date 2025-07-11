# uhop/core/ai_kernel_generator.py
import openai
import os
import re
import json
import time
import numpy as np
import hashlib
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from .tensor import Tensor
from .kernel_db import KernelDB
import shutil  # Added for file operations


class AIKernelGenerator:
    def __init__(self, api_key, backend='cuda'):
        """
        AI-powered kernel generator for U-HOP

        Args:
            api_key: OpenAI API key
            backend: Target backend ('cuda' or 'rocm')
        """
        openai.api_key = api_key
        self.backend = backend
        self.kernel_db = {}
        self.operation_signatures = {
            "matmul": {
                "params": ["const float* A", "const float* B", "float* C",
                           "int M", "int N", "int K"],
                "grid_block": "2D grid, 2D blocks"
            },
            "conv2d": {
                "params": ["const float* input", "const float* kernel", "float* output",
                           "int in_channels", "int in_height", "int in_width",
                           "int out_channels", "int kernel_size",
                           "int out_height", "int out_width",
                           "int stride", "int padding"],
                "grid_block": "3D grid, 3D blocks"
            },
            "relu": {
                "params": ["const float* input", "float* output", "int size"],
                "grid_block": "1D grid, 1D blocks"
            }
        }

    def refine_kernel(self, kernel_code, operation, constraints, input_shape, output_shape, error_log):
        """
        Refine an existing kernel using error feedback

        Args:
            kernel_code: Original kernel code
            error_log: Compilation error or validation failure details
        """
        prompt = f"""
            ## U-HOP Kernel Refinement Request
            We encountered an issue with this CUDA kernel and need your expertise to fix it.

            ### Problem Report:
            {error_log}

            ### Original Kernel:
            ```c
            {kernel_code}
            Constraints:
            {json.dumps(constraints, indent=2)}

            Instructions:
            Analyze the error and identify the root cause

            Fix the kernel while maintaining all optimizations

            Output ONLY the fixed kernel code in a code block
        """


        try:
            response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert CUDA debugger. Fix the provided kernel."
                },
                {
                    "role": "user",
                    "content": prompt
                }],
            temperature=0.1,  # Lower temp for debugging
            max_tokens=1800,
            top_p=0.9
            )

            fixed_code = response.choices[0].message.content.strip()
            return self._extract_kernel_code(fixed_code)

        except Exception as e:
            print(f"Refinement failed: {str(e)}")
            return None


    def get_or_generate_kernel(self, operation, constraints, input_shape, output_shape):
        """
            Smart kernel retrieval: use existing or generate new
        """
        # First try knowledge base
        best_kernel = self.kernel_db.get_best_kernel(operation, input_shape, output_shape)
        if best_kernel:
            return best_kernel["kernel_code"], best_kernel


        # Generate new kernel if not found
        kernel_code = self.generate_kernel(operation, constraints)
        return kernel_code, None


    def evaluate_and_refine(self, kernel_code, operation, constraints, input_shape, output_shape, max_attempts=3):
        """
        Evaluate kernel and refine if needed with error feedback loop
        """
        attempt = 0
        error_log = ""


        while attempt < max_attempts:
            attempt += 1
            print(f"Refinement attempt {attempt}/{max_attempts}")
            
            try:
                # Evaluate the kernel
                exec_time, kernel_path, error = self.evaluate_kernel(
                    kernel_code, operation, input_shape, output_shape
                )
                
                # Check if kernel is valid
                if error < 1e-3:  # Acceptable error threshold
                    # Save to knowledge base
                    self.kernel_db.add_kernel(
                        operation, constraints, kernel_code,
                        exec_time, error, input_shape, output_shape
                    )
                    return kernel_code, exec_time, error
                else:
                    error_log = f"Validation error: {error}"
                    
            except Exception as e:
                error_log = f"Compilation/execution error: {str(e)}"
            
            # Refine based on error
            kernel_code = self.refine_kernel(
                kernel_code, operation, constraints,
                input_shape, output_shape, error_log
            )
            
            if not kernel_code:
                break  # Refinement failed
                
        return None, float('inf'), float('inf')  # Failed after max attempts
    
    def _build_prompt(self, operation, constraints):
        """Create detailed prompt for AI code generation"""
        if operation not in self.operation_signatures:
            raise ValueError(f"Unsupported operation: {operation}")

        signature = self.operation_signatures[operation]
        
        prompt = f"""
            U-HOP AI Kernel Generation Request
            Operation: {operation}
            Hardware Target: {constraints.get('target_device', 'NVIDIA GPU')}
            Performance Goal: {constraints.get('goal', 'maximize throughput')}

            Constraints:
            {json.dumps(constraints, indent=2)}

            Required Optimizations:
            Shared memory tiling for data locality

            Coalesced global memory access patterns

            Minimize bank conflicts in shared memory

            Warp-level parallelism optimization

            {constraints.get('additional_optimizations', 'Loop unrolling where beneficial')}

            Efficient use of {constraints.get('registers', '32')} registers

            Kernel Requirements:
            Kernel name: {operation}_kernel

            Parameters: {', '.join(signature["params"])}

            Launch configuration: {signature["grid_block"]}

            Must include boundary checks for all memory accesses

            Must avoid thread divergence within warps

            Must use const correctness where appropriate

            Must include NaN and infinity checks in debug mode

            Output Format:

            // Optimized CUDA kernel for {operation}
            __global__ void {operation}_kernel({', '.join(signature["params"])}) {{
                // Implementation with detailed comments
            }}
            
            Example for matmul:

            __global__ void matmul_kernel(const float* A, const float* B, float* C, 
                                        int M, int N, int K) {{
                // Example implementation using shared memory tiles
            }}
            Note: Output ONLY the kernel code between triple backticks. No explanations.
        """
        return prompt.strip()


    def generate_kernel(self, operation, constraints):
        """Generate CUDA kernel using AI with rich prompt"""
        prompt = self._build_prompt(operation, constraints)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert CUDA kernel engineer. Output only complete, compilable kernel code."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.25,
                max_tokens=1800,
                top_p=0.95
            )
            
            kernel_code = response.choices[0].message.content.strip()
            return self._extract_kernel_code(kernel_code)
        
        except Exception as e:
            print(f"AI generation failed: {str(e)}")
            return None

    def _extract_kernel_code(self, raw_text):
        """Extract CUDA kernel code from AI response"""
        # Try to match code blocks with various CUDA markers
        patterns = [
            r'```[c|cuda|cc]*\n(.*?)```',
            r'__global__.*?\{.*?\}',
            r'// CUDA kernel(.*?)(?=\n```|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return raw_text.strip()

    def _validate_kernel_signature(self, kernel_code, operation):
        """Verify kernel has required signature"""
        required_name = f"{operation}_kernel"
        
        # Check for proper kernel declaration
        if f"__global__ void {required_name}(" not in kernel_code:
            # Attempt to fix common naming issues
            fixed_code = re.sub(
                r"__global__\s+void\s+\w+?\(", 
                f"__global__ void {required_name}(", 
                kernel_code
            )
            
            if f"__global__ void {required_name}(" not in fixed_code:
                raise ValueError(
                    f"Kernel must be named '{required_name}'. Found: {kernel_code[:200]}..."
                )
            return fixed_code
        
        return kernel_code

    def _add_safety_checks(self, kernel_code):
        """Inject boundary checks and NaN guards"""
        safety_code = """
        // U-HOP SAFETY CHECKS
        #ifdef UHOP_DEBUG
        #define CHECK_BOUNDS(idx, max) if ((idx) >= (max)) { return; }
        #define CHECK_NAN(value) if (isnan(value)) { return; }
        #else
        #define CHECK_BOUNDS(idx, max)
        #define CHECK_NAN(value)
        #endif
        """
        return safety_code + kernel_code

    def _build_kernel_arguments(self, operation, a, b, c, input_shape, output_shape):
        """Create kernel arguments based on operation type"""
        if operation == "matmul":
            M, K = input_shape
            _, N = output_shape
            return [
                a.data, b.data, c.data,
                np.int32(M), np.int32(N), np.int32(K)
            ]
        
        elif operation == "conv2d":
            batch, in_channels, in_h, in_w = input_shape
            out_channels, _, kernel_size, _ = b.shape
            padding = 1  # Default
            stride = 1   # Default
            
            out_h = (in_h - kernel_size + 2 * padding) // stride + 1
            out_w = (in_w - kernel_size + 2 * padding) // stride + 1
            
            return [
                a.data, b.data, c.data,
                np.int32(in_channels), 
                np.int32(in_h), np.int32(in_w),
                np.int32(out_channels), np.int32(kernel_size),
                np.int32(out_h), np.int32(out_w),
                np.int32(stride), np.int32(padding)
            ]
        
        elif operation == "relu":
            size = a.data.size
            return [a.data, c.data, np.int32(size)]
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _calculate_launch_params(self, operation, input_shape, output_shape):
        """Dynamically determine optimal launch configuration"""
        if operation == "matmul":
            M, _ = input_shape
            _, N = output_shape
            block = (16, 16, 1)
            grid_x = (N + block[0] - 1) // block[0]
            grid_y = (M + block[1] - 1) // block[1]
            return block, (grid_x, grid_y, 1)
        
        elif operation == "conv2d":
            _, out_channels, out_h, out_w = output_shape
            block = (8, 8, 4)
            grid_x = (out_w + block[0] - 1) // block[0]
            grid_y = (out_h + block[1] - 1) // block[1]
            grid_z = (out_channels + block[2] - 1) // block[2]
            return block, (grid_x, grid_y, grid_z)
        
        elif operation == "relu":
            size = input_shape[0] * input_shape[1]
            block = (256, 1, 1)
            grid = ((size + block[0] - 1) // block[0], 1, 1)
            return block, grid
        
        return (16, 16, 1), (1, 1, 1)  # Default

    def evaluate_kernel(self, kernel_code, operation, input_shape, output_shape):
        """
        Compile and benchmark AI-generated kernel
        
        Returns:
            (execution_time, kernel_path, validation_error)
        """
        if not kernel_code:
            return float('inf'), None, float('inf')
        
        try:
            # Validate and enhance kernel
            validated_code = self._validate_kernel_signature(kernel_code, operation)
            safeguarded_code = self._add_safety_checks(validated_code)
            
            # Create unique filename
            kernel_hash = hashlib.md5(safeguarded_code.encode()).hexdigest()[:8]
            kernel_filename = f"kernels/ai_{operation}_{kernel_hash}.cu"
            os.makedirs("kernels", exist_ok=True)
            
            # Save with metadata
            with open(kernel_filename, "w") as f:
                f.write(f"// U-HOP AI-Generated Kernel\n")
                f.write(f"// Operation: {operation}\n")
                f.write(f"// Input: {input_shape}, Output: {output_shape}\n")
                f.write(f"// Generated: {time.ctime()}\n\n")
                f.write(safeguarded_code)
            
            # Compile kernel
            mod = SourceModule(safeguarded_code)
            kernel_func = mod.get_function(f"{operation}_kernel")
            
            # Prepare test data
            a = Tensor(np.random.randn(*input_shape).astype(np.float32))
            b = Tensor(np.random.randn(*output_shape).astype(np.float32))
            c = Tensor(np.zeros(output_shape), dtype=np.float32)
            
            # Transfer to GPU
            a_gpu = a.gpu()
            b_gpu = b.gpu()
            c_gpu = c.gpu()
            
            # Get launch parameters
            block, grid = self._calculate_launch_params(operation, input_shape, output_shape)
            
            # Build arguments
            args = self._build_kernel_arguments(
                operation, a_gpu, b_gpu, c_gpu, input_shape, output_shape
            )
            
            # Warm-up
            kernel_func(*args, block=block, grid=grid)
            cuda.Context.synchronize()
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.perf_counter()
                kernel_func(*args, block=block, grid=grid)
                cuda.Context.synchronize()
                times.append(time.perf_counter() - start)
            
            # Get result and validate
            c_cpu = c_gpu.cpu().data
            ground_truth = self._compute_ground_truth(operation, a.data, b.data)
            error = np.max(np.abs(ground_truth - c_cpu))
            
            return np.median(times), kernel_filename, error
        
        except Exception as e:
            print(f"Kernel evaluation failed: {str(e)}")
            return float('inf'), kernel_filename, float('inf')

    def _compute_ground_truth(self, operation, a, b):
        """Compute reference output for validation"""
        if operation == "matmul":
            return np.matmul(a, b)
        elif operation == "conv2d":
            # Simplified CPU conv2d
            from scipy.signal import convolve2d
            _, in_channels, in_h, in_w = a.shape
            out_channels, _, kernel_size, _ = b.shape
            output = np.zeros((out_channels, in_h - kernel_size + 1, in_w - kernel_size + 1))
            
            for oc in range(out_channels):
                for ic in range(in_channels):
                    output[oc] += convolve2d(
                        a[0, ic], b[oc, ic], 
                        mode='valid'
                    )
            return output
        elif operation == "relu":
            return np.maximum(a, 0)
        else:
            return np.zeros_like(a)

    def evolve_kernel(self, operation, constraints, input_shape, output_shape, generations=5):
        """
        AI-driven kernel evolution
        
        Args:
            operation: Kernel operation (matmul, conv2d, relu)
            constraints: Optimization constraints dictionary
            input_shape: Shape of input tensor
            output_shape: Shape of output tensor
            generations: Number of evolution iterations
        
        Returns:
            (best_kernel_path, best_time, evolution_history)
        """
        best_time = float('inf')
        best_kernel = None
        history = []
        
        print(f"\n🚀 Starting AI Kernel Evolution for {operation}")
        print(f"Input: {input_shape}, Output: {output_shape}")
        print(f"Constraints: {json.dumps(constraints, indent=2)}")
        
        for gen in range(generations):
            print(f"\n🌀 Generation {gen+1}/{generations}")
            
            # Generate kernel variation
            kernel_code = self.generate_kernel(operation, constraints)
            if not kernel_code:
                print("⚠️  Skipping generation due to AI failure")
                continue
                
            # Evaluate kernel
            start_time = time.perf_counter()
            exec_time, kernel_path, error = self.evaluate_kernel(
                kernel_code, operation, input_shape, output_shape
            )
            eval_time = time.perf_counter() - start_time
            
            # Record results
            result = {
                "generation": gen+1,
                "exec_time": exec_time,
                "validation_error": error,
                "kernel_path": kernel_path,
                "eval_time": eval_time,
                "constraints": constraints,
                "timestamp": time.time()
            }
            history.append(result)
            
            # Print results
            status = "✅" if error < 1e-5 else "⚠️"
            print(f"{status} Kernel {gen+1} | "
                f"Time: {exec_time:.6f}s | "
                f"Error: {error:.4e} | "
                f"Eval: {eval_time:.2f}s")
            
            # Update best kernel
            if exec_time < best_time and error < 1e-3:
                best_time = exec_time
                best_kernel = kernel_path
                print(f"🏆 NEW BEST: {kernel_path} ({best_time:.6f}s)")
                
                # Save best kernel
                best_dir = "kernels/best"
                os.makedirs(best_dir, exist_ok=True)
                shutil.copy(kernel_path, os.path.join(best_dir, f"best_{operation}.cu"))
        
        # Save evolution history
        history_file = f"kernel_evolution_{operation}_{int(time.time())}.json"
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\n🔥 Evolution complete! Best time: {best_time:.6f}s")
        return best_kernel, best_time, history

    def optimize_workload(self, operation, problem_sizes, constraints, generations=3):
        """
        Optimize kernel across multiple problem sizes
        
        Args:
            operation: Kernel operation
            problem_sizes: List of (input_shape, output_shape) tuples
            constraints: Optimization constraints
            generations: Generations per problem size
        
        Returns:
            Dictionary of best kernels for each size
        """
        results = {}
        
        for i, (input_shape, output_shape) in enumerate(problem_sizes):
            print(f"\n🔧 Optimizing {operation} for size {i+1}/{len(problem_sizes)}")
            print(f"Input: {input_shape}, Output: {output_shape}")
            
            best_kernel, best_time, _ = self.evolve_kernel(
                operation=operation,
                constraints=constraints,
                input_shape=input_shape,
                output_shape=output_shape,
                generations=generations
            )
            
            results[json.dumps(input_shape)] = {
                "kernel": best_kernel,
                "time": best_time
            }
        
        return results