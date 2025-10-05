# uhop/core/ai_kernel_optimizer.py

import os
import time
import numpy as np
import re
from pathlib import Path
from .tensor import Tensor
from .tuner import KernelTuner
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

class AIKernelOptimizer:
    def __init__(self, ai_client, operation, constraints, max_generations=5):
        self.ai = ai_client
        self.operation = operation
        self.constraints = constraints
        self.max_generations = max_generations
        self.best_kernel = None
        self.best_time = float('inf')
        self.best_code = None
        self.kernel_db = {}
        Path("kernels").mkdir(exist_ok=True)

    def _build_prompt(self, feedback=None):
        base_prompt = f"""
You are an expert CUDA programmer. Generate a correct and optimized CUDA kernel for {self.operation}.
Constraints: {self.constraints}

Requirements:
- Use shared memory and tiling.
- Minimize global memory reads/writes.
- Kernel function name: {self.operation}_kernel
- Output only valid CUDA code between triple backticks.
"""
        if feedback:
            base_prompt += f"\nFix this issue:\n{feedback}\n"
        return base_prompt

    def _extract_code(self, text):
        match = re.search(r'```cuda\n(.*?)```', text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def _validate(self, cpu_result, gpu_result, tolerance=1e-4):
        return np.allclose(cpu_result, gpu_result, atol=tolerance), np.max(np.abs(cpu_result - gpu_result))

    def _compile_and_run(self, code, input_shape, block=(16,16,1)):
        mod = SourceModule(code)
        kernel_func = mod.get_function(f"{self.operation}_kernel")

        a_np = np.random.randn(*input_shape).astype(np.float32)
        b_np = np.random.randn(input_shape[1], input_shape[0]).astype(np.float32)

        a = Tensor(a_np).gpu()
        b = Tensor(b_np).gpu()
        c = Tensor(np.zeros((input_shape[0], input_shape[0]), dtype=np.float32)).gpu()

        cpu_c = np.matmul(a_np, b_np)

        grid = ((input_shape[0] + block[0] - 1) // block[0],) * 2

        times = []
        for _ in range(5):
            start = time.perf_counter()
            kernel_func(a.data, b.data, c.data, np.int32(input_shape[0]), np.int32(input_shape[0]), np.int32(input_shape[1]), block=block, grid=grid)
            cuda.Context.synchronize()
            times.append(time.perf_counter() - start)

        gpu_c = c.cpu().data

        valid, max_diff = self._validate(cpu_c, gpu_c)
        avg_time = np.mean(times)
        return valid, max_diff, avg_time

    def optimize(self, input_shape=(128,128)):
        for gen in range(1, self.max_generations + 1):
            prompt = self._build_prompt()
            response = self.ai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2
            )
            code = self._extract_code(response.choices[0].message.content)
            valid, diff, exec_time = self._compile_and_run(code, input_shape)

            if not valid:
                feedback = f"Max difference: {diff}. Kernel is incorrect."
                prompt = self._build_prompt(feedback)
                continue

            if exec_time < self.best_time:
                self.best_time = exec_time
                self.best_kernel = code
                self.best_code = code

            print(f"Gen {gen}: Valid: {valid} Time: {exec_time:.6f}s Best: {self.best_time:.6f}s")

        if self.best_kernel:
            filename = f"kernels/best_{self.operation}_{int(time.time())}.cu"
            with open(filename, 'w') as f:
                f.write(self.best_kernel)
            print(f"Best kernel saved: {filename}")

        return self.best_kernel, self.best_time
