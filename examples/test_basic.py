import sys
import os
from pathlib import Path

# Add the parent directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


print("Testing basic imports...")
try:
    from uhop import Tensor, Executor
    from uhop.adapters import CUDA_AVAILABLE
    print("✓ Imports successful")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    
    # Test basic tensor operations
    import numpy as np
    a = Tensor(np.random.rand(5, 5))
    b = Tensor(np.random.rand(5, 5))
    print("✓ Tensor creation successful")
    
    # Test executor
    executor = Executor('cpu')
    result = executor.matmul(a, b)
    print("✓ Matmul operation successful")
    print(f"Result shape: {result.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()