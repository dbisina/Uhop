# uhop/adapters/cpu_adapter.py
import numpy as np
from ..core.tensor import Tensor

def matmul(a, b):
    a_cpu = a.data if a.device == 'cpu' else a.cpu().data
    b_cpu = b.data if b.device == 'cpu' else b.cpu().data
    result = np.matmul(a_cpu, b_cpu)
    return Tensor(result, device='cpu')