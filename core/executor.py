# uhop/core/executor.py
from uhop.adapters import cpu_adapter, cuda_adapter
from uhop.core.monitor import timed_operation


class Executor:
    def __init__(self, backend='cpu'):
        self.backend = backend

    @timed_operation
    def matmul(self, a, b):
        if self.backend == 'cpu':
            a, b = a.cpu(), b.cpu()
            return cpu_adapter.matmul(a, b)
        elif self.backend == 'cuda':
            a, b = a.gpu(), b.gpu()
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
            return cuda_adapter.relu(a)

    @timed_operation
    def conv2d(self, input, kernel, stride=1, padding=0):
        if self.backend == 'cpu':
            input = input.cpu()
            kernel = kernel.cpu()
            return cpu_adapter.conv2d(input, kernel, stride, padding)
        elif self.backend == 'cuda':
            input = input.gpu()
            kernel = kernel.gpu()
            return cuda_adapter.conv2d(input, kernel, stride, padding)