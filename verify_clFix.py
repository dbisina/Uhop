import shutil

# Check if cl.exe is available
cl_path = shutil.which('cl.exe')
if cl_path:
    print(f"✓ cl.exe found at: {cl_path}")
else:
    print("✗ cl.exe not found in PATH")
    
# Check if nvcc is available
nvcc_path = shutil.which('nvcc')
if nvcc_path:
    print(f"✓ nvcc found at: {nvcc_path}")
else:
    print("✗ nvcc not found in PATH")