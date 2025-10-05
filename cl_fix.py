import os
import subprocess

# Function to find cl.exe
def find_cl_exe():
    # Common installation paths for Visual Studio Build Tools
    search_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    ]
    
    for base_path in search_paths:
        if os.path.exists(base_path):
            # Look for MSVC compiler in the expected subdirectory
            msvc_path = os.path.join(base_path, "VC", "Tools", "MSVC")
            if os.path.exists(msvc_path):
                # Find the latest version
                versions = [d for d in os.listdir(msvc_path) if os.path.isdir(os.path.join(msvc_path, d))]
                if versions:
                    latest_version = sorted(versions)[-1]  # Get the latest version
                    cl_path = os.path.join(msvc_path, latest_version, "bin", "Hostx64", "x64")
                    if os.path.exists(os.path.join(cl_path, "cl.exe")):
                        return cl_path
    
    return None

# Find and add cl.exe to PATH
cl_path = find_cl_exe()
if cl_path:
    print(f"Found cl.exe at: {cl_path}")
    os.environ['PATH'] = cl_path + ';' + os.environ['PATH']
    print("Added cl.exe to PATH")
else:
    print("Could not find cl.exe. Please check your Visual Studio Build Tools installation.")
    
# Also add Windows SDK paths if needed
sdk_paths = [
    r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64",
    r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64",
]

for path in sdk_paths:
    if os.path.exists(path):
        os.environ['PATH'] = path + ';' + os.environ['PATH']
        print(f"Added Windows SDK to PATH: {path}")

# Verify the setup
import shutil
cl_path = shutil.which('cl.exe')
if cl_path:
    print(f"✓ cl.exe now found at: {cl_path}")
else:
    print("✗ cl.exe still not found in PATH")