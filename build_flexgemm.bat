@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set TORCH_CUDA_ARCH_LIST=12.0
C:\workspace\MODEL\TRELLIS.2\.venv\Scripts\pip.exe install C:\Users\kschmid\AppData\Local\Temp\trellis2_extensions\FlexGEMM --no-build-isolation --force-reinstall --no-deps > C:\workspace\flexgemm_build.log 2>&1
echo Exit code: %ERRORLEVEL% >> C:\workspace\flexgemm_build.log
