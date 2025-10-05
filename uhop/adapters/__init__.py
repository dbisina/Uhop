from . import cpu_adapter
try:
    from . import cuda_adapter
except ImportError:
    print("CUDA not available, CPU-only mode")
    cuda_adapter = None