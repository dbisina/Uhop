import os
import sys

# Fix the CUDA path issue
cuda_path = os.environ.get('CUDA_PATH', '')
if cuda_path and 'Program Files' in cuda_path and not cuda_path.startswith('C:\\'):
    # Fix the path format
    correct_path = 'C:\\' + cuda_path.split(':', 1)[-1].lstrip('\\')
    os.environ['CUDA_PATH'] = correct_path
    print(f"Fixed CUDA_PATH: {correct_path}")
    
    # Add to DLL path
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(os.path.join(correct_path, 'bin'))
    
    # Add to system path
    sys.path.insert(0, os.path.join(correct_path, 'bin'))