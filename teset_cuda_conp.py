import os
import subprocess
import sys

# First, let's verify cl.exe is accessible
def test_cl_exe():
    try:
        result = subprocess.run(['cl.exe'], capture_output=True, text=True)
        if result.returncode != 0:
            # This is expected - cl.exe should show usage info when run without args
            print("✓ cl.exe is accessible (shows usage info when run without args)")
            return True
        else:
            print("✓ cl.exe is accessible")
            return True
    except Exception as e:
        print(f"✗ cl.exe is not accessible: {e}")
        return False

# Test nvcc compilation
def test_nvcc_compilation():
    try:
        # Create a simple CUDA test file
        cuda_test_code = """
#include <stdio.h>

__global__ void hello_cuda() {
    printf("Hello from CUDA!\\n");
}

int main() {
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
"""
        
        with open('test_cuda.cu', 'w') as f:
            f.write(cuda_test_code)
        
        # Try to compile with nvcc
        result = subprocess.run([
            'nvcc', 'test_cuda.cu', 
            '-o', 'test_cuda.exe',
            '-arch=sm_89'  # Use your GPU's compute capability (8.9 for RTX 4060)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ nvcc compilation successful")
            
            # Try to run the compiled program
            run_result = subprocess.run(['test_cuda.exe'], capture_output=True, text=True)
            if run_result.returncode == 0:
                print("✓ CUDA program runs successfully")
                return True
            else:
                print(f"✗ CUDA program failed to run: {run_result.stderr}")
                return False
        else:
            print(f"✗ nvcc compilation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error during nvcc test: {e}")
        return False
    finally:
        # Clean up
        for f in ['test_cuda.cu', 'test_cuda.exe']:
            if os.path.exists(f):
                os.remove(f)

# Test PyCUDA compilation
def test_pycuda_compilation():
    try:
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        
        # Simple CUDA kernel
        kernel_code = """
__global__ void multiply(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
"""
        
        # Try to compile
        mod = SourceModule(kernel_code)
        print("✓ PyCUDA compilation successful")
        return True
        
    except Exception as e:
        print(f"✗ PyCUDA compilation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing CUDA compilation setup...")
    
    # Test cl.exe accessibility
    cl_ok = test_cl_exe()
    
    # Test nvcc compilation
    nvcc_ok = test_nvcc_compilation()
    
    # Test PyCUDA compilation
    pycuda_ok = test_pycuda_compilation()
    
    print("\n" + "="*50)
    if cl_ok and nvcc_ok and pycuda_ok:
        print("✓ All tests passed! CUDA compilation is working correctly.")
        print("You can now run the U-HOP AI kernel demo.")
    else:
        print("✗ Some tests failed. Please check your setup.")
        
        if not cl_ok:
            print("- cl.exe is not accessible")
        if not nvcc_ok:
            print("- nvcc compilation failed")
        if not pycuda_ok:
            print("- PyCUDA compilation failed")