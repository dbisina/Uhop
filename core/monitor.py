import time
from collections import defaultdict
from .tensor import Tensor


class PerformanceMonitor:
    def __init__(self):
        self.records = defaultdict(list)
        self.enabled = True
    
    def record(self, operation, backend, duration):
        if self.enabled:
            self.records[(operation, backend)].append(duration)
    
    def get_stats(self, operation, backend):
        durations = self.records.get((operation, backend), [])
        if not durations:
            return None
        return {
            'count': len(durations),
            'total': sum(durations),
            'avg': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations)
        }
    
    def reset(self):
        self.records.clear()


# Global instance
monitor = PerformanceMonitor()


def timed_operation(func):
    """Decorator for automatic performance monitoring"""
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
        
        monitor.record(operation, backend, duration)
        return result
    return wrapper