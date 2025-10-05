import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

print("Checking CUDA installation...")

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    print("✓ pycuda imported successfully")
    
    # Check CUDA device
    device = cuda.Device(0)
    print(f"✓ CUDA Device: {device.name()}")
    print(f"✓ Compute Capability: {device.compute_capability()}")
    
    # Check CUDA path
    cuda_path = os.environ.get('CUDA_PATH', '')
    print(f"✓ CUDA_PATH: {cuda_path}")
    
except Exception as e:
    print(f"✗ CUDA error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you have an NVIDIA GPU")
    print("2. Install NVIDIA drivers")
    print("3. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
    print("4. Install pycuda: pip install pycuda")