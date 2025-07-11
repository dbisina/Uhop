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
        
        # Calculate FLOPs if possible
        flops = None
        if operation == "matmul":
            a, b = args[1], args[2]  # self, a, b
            M, K = a.shape
            _, N = b.shape
            flops = 2 * M * N * K  # FLOP count for matmul
        
        monitor.record(operation, backend, duration, flops=flops)
        return result
    return wrapper