@echo off
echo Setting up Visual Studio environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo Setting up CUDA environment...
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_PATH%\bin;%PATH%

echo Running U-HOP AI kernel demo...
python "C:\Users\dicks\Downloads\Uhop\Uhop\examples\ai_kernel_demo.py"

pause