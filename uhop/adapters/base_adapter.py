# uhop/adapters/base_adapter.py
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @abstractmethod
    def matmul(self, a, b):
        pass
    
    @abstractmethod
    def conv2d(self, input, kernel, stride=1, padding=0):
        pass
    
    @abstractmethod
    def relu(self, a):
        pass
    
    @staticmethod
    def create(backend='cpu'):
        if backend == 'cuda':
            from .cuda_adapter import CudaAdapter
            return CudaAdapter()
        elif backend == 'rocm':
            from .rocm_adapter import RocmAdapter
            return RocmAdapter()
        else:
            from .cpu_adapter import CpuAdapter
            return CpuAdapter()