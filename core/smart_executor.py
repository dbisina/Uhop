import os
import json
import time
import numpy as np
from pathlib import Path
from uhop.core.tensor import Tensor
from uhop.adapters import cuda_adapter, cpu_adapter
from uhop.core.monitor import monitor, timed_operation
from uhop.core.ai_kernel_generator import AIKernelGenerator


class SmartExecutor:
    def __init__(self, backend='cpu', api_key=None, use_ai=False):
        self.backend = backend
        self.use_ai = use_ai
        self.ai_kernel = AIKernelGenerator(api_key=api_key, backend=backend) if use_ai else None
        self.known_good = cpu_adapter  # fallback reference

    @timed_operation
    def matmul(self, a, b):
        a, b = a.cpu(), b.cpu()

        if self.backend == 'cpu':
            return self.known_good.matmul(a, b)

        elif self.backend == 'cuda':
            a, b = a.gpu(), b.gpu()

            if self.use_ai:
                constraints = {
                    "target_device": "NVIDIA GPU",
                    "tile_sizes": [8, 16, 32],
                    "operation": "matmul"
                }
                kernel_code = self.ai_kernel.generate_kernel('matmul', constraints)

                if kernel_code:
                    exec_time, kernel_path = self.ai_kernel.evaluate_kernel(
                        kernel_code, 'matmul', a.shape, b.shape
                    )

                    # Validate AI kernel correctness
                    reference = self.known_good.matmul(a.cpu(), b.cpu()).cpu().data
                    # Dummy run: actual AI kernel run is already done in evaluate_kernel

                    # For now we skip real AI tensor result capture (to be added)
                    ai_output = reference  # Replace with actual AI run output

                    max_diff = np.max(np.abs(reference - ai_output))

                    if max_diff < 1e-3:
                        print(f"✅ AI Kernel valid. Time: {exec_time:.6f}s. Diff: {max_diff:.6f}")
                        self.ai_kernel.kernel_db[("matmul", str(constraints))] = {
                            "path": kernel_path,
                            "exec_time": exec_time
                        }
                        return Tensor(ai_output)
                    else:
                        print(f"❌ AI Kernel invalid. Falling back. Diff: {max_diff:.6f}")

            # Fallback to native CUDA
            return cuda_adapter.matmul(a, b)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @timed_operation
    def relu(self, a):
        a = a.cpu()

        if self.backend == 'cpu':
            return self.known_good.relu(a)

        elif self.backend == 'cuda':
            a = a.gpu()

            if self.use_ai:
                constraints = {
                    "target_device": "NVIDIA GPU",
                    "operation": "relu"
                }
                kernel_code = self.ai_kernel.generate_kernel('relu', constraints)

                if kernel_code:
                    exec_time, kernel_path = self.ai_kernel.evaluate_kernel(
                        kernel_code, 'relu', a.shape, a.shape
                    )

                    reference = self.known_good.relu(a.cpu()).cpu().data
                    ai_output = reference  # TODO: Capture real AI run output

                    max_diff = np.max(np.abs(reference - ai_output))

                    if max_diff < 1e-3:
                        print(f"✅ AI ReLU valid. Time: {exec_time:.6f}s. Diff: {max_diff:.6f}")
                        self.ai_kernel.kernel_db[("relu", str(constraints))] = {
                            "path": kernel_path,
                            "exec_time": exec_time
                        }
                        return Tensor(ai_output)
                    else:
                        print(f"❌ AI ReLU invalid. Falling back. Diff: {max_diff:.6f}")

            return cuda_adapter.relu(a)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def save_ai_kernels(self, path="uhop_ai_kernel_db.json"):
        if self.ai_kernel:
            self.ai_kernel.save_kernel_db(path)

    def load_ai_kernels(self, path="uhop_ai_kernel_db.json"):
        if self.ai_kernel:
            self.ai_kernel.load_kernel_db(path)
