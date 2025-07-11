# uhop/core/tensor.py
import numpy as np
import pycuda.gpuarray as gpuarray

class Tensor:
    def __init__(self, data, device='cpu'):
        self.data = data
        self.device = device

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype if self.device == 'cpu' else self.data.dtype

    def cpu(self):
        if self.device == 'cpu':
            return self
        return Tensor(self.data.get(), device='cpu')

    def gpu(self):
        if self.device == 'gpu':
            return self
        return Tensor(gpuarray.to_gpu(self.data), device='gpu')

    def __repr__(self):
        return f"Tensor(shape={self.shape}, device='{self.device}')"