# uhop/core/executor.py
from ..adapters import cpu_adapter, cuda_adapter

class Executor:
    def __init__(self, backend='cpu'):
        self.backend = backend

    def matmul(self, a, b):
        if self.backend == 'cpu':
            a, b = a.cpu(), b.cpu()
            return cpu_adapter.matmul(a, b)
        elif self.backend == 'cuda':
            a, b = a.gpu(), b.gpu()
            return cuda_adapter.matmul(a, b)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")