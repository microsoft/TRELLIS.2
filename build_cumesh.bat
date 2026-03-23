@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set TORCH_CUDA_ARCH_LIST=12.0
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set EXT_DIR=%TEMP%\trellis2_extensions
mkdir "%EXT_DIR%" 2>nul
if not exist "%EXT_DIR%\CuMesh" (
    git clone https://github.com/JeffreyXiang/CuMesh.git "%EXT_DIR%\CuMesh" --recursive
)
C:\workspace\MODEL\TRELLIS.2\.venv\Scripts\pip install "%EXT_DIR%\CuMesh" --no-build-isolation --force-reinstall
