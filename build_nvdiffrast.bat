@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set TORCH_CUDA_ARCH_LIST=12.0
c:\workspace\MODEL\TRELLIS.2\.venv\Scripts\pip install --no-build-isolation --force-reinstall git+https://github.com/NVlabs/nvdiffrast.git
pause
