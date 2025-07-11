# uhop/core/tuner.py
import time
import numpy as np
import pycuda.driver as cuda
from .tensor import Tensor

class KernelTuner:
    def __init__(self, backend='cuda'):
        self.backend = backend
        self.optimal_configs = {}  # {(operation, shape): (block_x, block_y)}
    
    def tune_matmul(self, a, b, block_sizes=[16, 32, 64]):
        try:
            """Auto-tune matrix multiplication kernel"""
            from .executor import Executor
            
            # Warm-up run
            executor = Executor(self.backend)
            _ = executor.matmul(a, b)
            
            # Benchmark different block sizes
            best_time = float('inf')
            best_config = None
            
            for block_size in block_sizes:
                times = []
                for _ in range(5):  # Multiple runs
                    start = time.perf_counter()
                    result = self._run_with_block_size(a, b, block_size)
                    if self.backend == 'cuda':
                        cuda.Context.synchronize()
                    times.append(time.perf_counter() - start)
                
                avg_time = np.mean(times)
                if avg_time < best_time:
                    best_time = avg_time
                    best_config = block_size
            
            # Store optimal configuration
            key = ('matmul', a.shape, b.shape)
            self.optimal_configs[key] = best_config
            return best_config
        except Exception as e:
            print(f"Error during tuning: {e}")
            return 32
    
    def _run_with_block_size(self, a, b, block_size):
        """Execute matmul with specific block size"""
        if self.backend == 'cuda':
            from uhop.adapters import cuda_adapter
            return cuda_adapter.matmul(a, b, block_size=block_size)
        else:
            from uhop.adapters import cpu_adapter
            return cpu_adapter.matmul(a, b)
    
    def get_optimal_block_size(self, operation, a_shape, b_shape=None):
        """Retrieve cached optimal configuration"""
        key = (operation, a_shape, b_shape)
        return self.optimal_configs.get(key, 32)  # Default to 32