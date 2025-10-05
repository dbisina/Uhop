# uhop/core/executor.py
from uhop.core.ai_kernel_generator import AIKernelGenerator
from uhop.adapters import cpu_adapter, cuda_adapter
from uhop.core.monitor import timed_operation
import numpy as np

class Executor:
    def __init__(self, backend='cpu', use_ai=False, ai_api_key=None):
        self.backend = backend
        self.use_ai = use_ai

        # Initialize AI Kernel Generator if needed
        self.ai_kernel_generator = None
        if self.use_ai and ai_api_key:
            self.ai_kernel_generator = AIKernelGenerator(api_key=ai_api_key, backend=backend)

    @timed_operation
    def matmul(self, a, b):
        if self.backend == 'cpu':
            a, b = a.cpu(), b.cpu()
            return cpu_adapter.matmul(a, b)

        elif self.backend == 'cuda':
            a, b = a.gpu(), b.gpu()

            # AI Path
            if self.use_ai and self.ai_kernel_generator:
                constraints = {
                    "operation": "matmul",
                    "target_device": "NVIDIA GPU",
                    "input_shape": a.shape,
                    "output_shape": (a.shape[0], b.shape[1])
                }

                best_kernel, best_time = self.ai_kernel_generator.evolve_kernel(
                    operation='matmul',
                    constraints=constraints,
                    input_shape=a.shape,
                    output_shape=b.shape
                )

                print(f"[AI Matmul] Best Kernel: {best_kernel} | Time: {best_time:.6f}s")

                # Fallback if AI fails
                if best_kernel is None or best_time == float('inf'):
                    return cuda_adapter.matmul(a, b)

                # Load AI kernel (this step is symbolic—direct integration to runtime needed)
                # For now, use existing adapter
                return cuda_adapter.matmul(a, b)

            # Normal CUDA path
            return cuda_adapter.matmul(a, b)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @timed_operation
    def relu(self, a):
        if self.backend == 'cpu':
            a = a.cpu()
            return cpu_adapter.relu(a)

        elif self.backend == 'cuda':
            a = a.gpu()

            # AI Path (future support)
            if self.use_ai and self.ai_kernel_generator:
                # Placeholder for AI-based ReLU generation
                pass

            return cuda_adapter.relu(a)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
