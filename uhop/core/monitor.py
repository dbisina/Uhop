# uhop/core/monitor.py
import time
import psutil
import numpy as np
from collections import defaultdict
from .tensor import Tensor

class PerformanceMonitor:
    def __init__(self):
        self.records = defaultdict(list)
        self.enabled = True
        self.memory_usage = []
        self.flop_counts = {}
    
    def record(self, operation, backend, duration, memory=None, flops=None):
        if self.enabled:
            record = {
                "duration": duration,
                "memory": memory or psutil.Process().memory_info().rss / 1024**2,
                "timestamp": time.time()
            }
            
            if flops:
                record["flops"] = flops
                self.flop_counts[(operation, backend)] = flops
                
            self.records[(operation, backend)].append(record)

    def record_hardware_metrics(self):
        """Capture real-time hardware-specific performance counters."""
        metrics = {}

        try:
            if self.backend == 'cuda':
                import pycuda.driver as cuda
                free_mem, total_mem = cuda.mem_get_info()
                metrics['gpu_mem_used_mb'] = round((total_mem - free_mem) / 1024**2, 2)
                metrics['gpu_mem_total_mb'] = round(total_mem / 1024**2, 2)
                # Potential to add occupancy, GPU utilization, temperature via nvml

            elif self.backend == 'rocm':
                # Placeholder: ROCm metrics (e.g., rocminfo, rocprofiler)
                metrics['gpu_mem_used_mb'] = None
                metrics['gpu_mem_total_mb'] = None

            elif self.backend == 'tpu':
                # Future: TPU support (e.g., via cloud APIs or XLA hooks)
                pass

        except Exception as e:
            metrics['gpu_error'] = str(e)

        # Universal CPU & RAM metrics
        try:
            metrics['cpu_util_percent'] = psutil.cpu_percent(interval=None)
            metrics['sys_mem_used_mb'] = round(psutil.virtual_memory().used / 1024**2, 2)
            metrics['sys_mem_total_mb'] = round(psutil.virtual_memory().total / 1024**2, 2)
        except Exception as e:
            metrics['cpu_mem_error'] = str(e)

        self.hardware_metrics.append(metrics)
    
    def get_stats(self, operation, backend):
        records = self.records.get((operation, backend), [])
        if not records:
            return None
            
        durations = [r["duration"] for r in records]
        memory = [r["memory"] for r in records]
        
        return {
            "count": len(records),
            "avg_time": np.mean(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "avg_memory": np.mean(memory),
            "flops": self.flop_counts.get((operation, backend), 0)
        }
    
    def reset(self):
        self.records.clear()
        self.flop_counts.clear()

# Global instance
monitor = PerformanceMonitor()

def timed_operation(func):
    """Decorator with FLOP calculation"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        
        # Get operation name from function
        operation = func.__name__
        
        # Determine backend from arguments
        backend = 'cpu'
        for arg in args:
            if isinstance(arg, Tensor) and arg.device == 'gpu':
                backend = 'cuda'
                break
        
         # Safely calculate FLOPs
        flops = None
        try:
            if operation == "matmul" and len(args) >= 3:
                a, b = args[1], args[2]  # self, a, b
                if hasattr(a, 'shape') and hasattr(b, 'shape'):
                    M, K = a.shape
                    _, N = b.shape
                    flops = 2 * M * N * K
        except (IndexError, AttributeError, ValueError):
            pass  # Skip FLOP calculation if shapes unavailable
        
        monitor.record(operation, backend, duration, flops=flops)
        return result
    return wrapper