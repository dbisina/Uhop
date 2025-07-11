# uhop/adapters/cpu_adapter.py
import numpy as np
from uhop.core.tensor import Tensor


def matmul(a, b):
    a_cpu = a.data if a.device == 'cpu' else a.cpu().data
    b_cpu = b.data if b.device == 'cpu' else b.cpu().data
    result = np.matmul(a_cpu, b_cpu)
    return Tensor(result, device='cpu')


def relu(a):
    a_cpu = a.data if a.device == 'cpu' else a.cpu().data
    result = np.maximum(a_cpu, 0)
    return Tensor(result, device='cpu')


def conv2d(input, kernel, stride=1, padding=0):
    input_cpu = input.data if input.device == 'cpu' else input.cpu().data
    kernel_cpu = kernel.data if kernel.device == 'cpu' else kernel.cpu().data
    
    from scipy.signal import convolve2d
    batch, in_channels, in_h, in_w = input_cpu.shape
    out_channels, _, kernel_h, kernel_w = kernel_cpu.shape
    
    # Calculate output dimensions
    out_h = (in_h - kernel_h + 2 * padding) // stride + 1
    out_w = (in_w - kernel_w + 2 * padding) // stride + 1
    
    output = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float32)
    
    for b in range(batch):
        for c_out in range(out_channels):
            for c_in in range(in_channels):
                input_slice = input_cpu[b, c_in]
                kernel_slice = kernel_cpu[c_out, c_in]
                
                # Apply convolution
                convolved = convolve2d(
                    input_slice, kernel_slice, 
                    mode='same' if padding else 'valid'
                )
                
                # Stride handling
                if stride > 1:
                    convolved = convolved[::stride, ::stride]
                
                # Accumulate results
                output[b, c_out] += convolved[:out_h, :out_w]
    
    return Tensor(output, device='cpu')