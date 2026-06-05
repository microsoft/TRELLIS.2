@echo off
cd /d %~dp0
Title Trellis2 BUILD (no wheels) (WINDOWS + Python 3.13 + Torch 2.9 + Cuda 13)

setlocal EnableExtensions EnableDelayedExpansion
call :set_colors

echo %green%=====================================%reset%
echo %green%   Building TRELLIS.2 (WINDOWS)     %reset%
echo %green%=====================================%reset%
echo.

:: -------------------------------------------------
:: Resolve Python
:: -------------------------------------------------
for %%P in (python.exe) do set "PYTHON_PATH=%%~$PATH:P"

if not defined PYTHON_PATH (
    echo %red%ERROR: Python not found in PATH%reset%
    echo Install Python from https://www.python.org
    exit /b 1
)

echo Using Python:
%PYTHON_PATH% --version
echo Location: %PYTHON_PATH%
echo.

:: -------------------------------------------------
:: Version checks
:: -------------------------------------------------
call :get_versions

if not "%PYTHON_VERSION%"=="3.13" (
    echo %red%ERROR: Python 3.13 is required%reset%
    exit /b 1
)

if not "%TORCH_VERSION%"=="2.9" (
    echo %red%ERROR: Torch 2.9.x required%reset%
    echo Detected Torch: %TORCH_VERSION%
    exit /b 1
)

if not "%CUDA_VERSION%"=="13.0" (
    echo %red%ERROR: CUDA 13.0 required%reset%
    echo Detected CUDA: %CUDA_VERSION%
    exit /b 1
)

:: -------------------------------------------------
:: Preconditions
:: -------------------------------------------------
echo Assumptions:
echo - Torch 2.9.1 + cu130 already installed
echo - CUDA Toolkit 13.0 installed (nvcc available)
echo - VC++ Redistributable installed
echo.

where nvcc
if errorlevel 1 (
    echo %yellow%WARNING: nvcc not found in PATH%reset%
    echo FlashAttention build may fail
)
echo.

:: ===============================
:: MSVC (Visual Studio Build Tools)
:: ===============================
echo.
echo ===============================================
echo   Configuring MSVC toolchain
echo ===============================================
echo.

set "MSVC_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Using MSVC environment script:
echo   %MSVC_VCVARS%
echo.

if not exist "%MSVC_VCVARS%" (
    echo ERROR: MSVC vcvars64.bat not found
    echo.
    echo Expected location:
    echo   %MSVC_VCVARS%
    echo.
    echo If your Visual Studio is installed elsewhere, edit this path in the script.
    echo Required components:
    echo   - Desktop development with C++
    echo   - MSVC v143 build tools
    echo   - Windows 10/11 SDK
    echo.
    exit /b 1
)

call "%MSVC_VCVARS%"

:: ===============================
:: BUILD FLAGS
:: ===============================
set DISTUTILS_USE_SDK=1
set MSSdk=1
set USE_NINJA=1
set MAX_JOBS=16
set CMAKE_BUILD_PARALLEL_LEVEL=16

echo.
echo Building with:
echo   DISTUTILS_USE_SDK=%DISTUTILS_USE_SDK%
echo   MSSdk=%MSSdk%
echo   USE_NINJA=%USE_NINJA%
echo   MAX_JOBS=%MAX_JOBS%
echo   CMAKE_BUILD_PARALLEL_LEVEL=%CMAKE_BUILD_PARALLEL_LEVEL%
echo.

:: -------------------------------------------------
:: CUDA runtime DLL visibility (safe)
:: -------------------------------------------------
if defined CUDA_PATH call set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

:: ===============================
:: TOOLS
:: ===============================
%PYTHON_PATH% -m pip install -U setuptools wheel ninja cmake pybind11 packaging

:: ===============================
:: RUNTIME DEPS
:: ===============================
%PYTHON_PATH% -m pip install ^
  trimesh ^
  shapely ^
  rtree ^
  easydict ^
  plyfile ^
  zstandard

:: =================================================
:: nvdiffrast
:: =================================================
echo.
echo %green%Building nvdiffrast...%reset%

%PYTHON_PATH% -m pip uninstall -y nvdiffrast
%PYTHON_PATH% -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation --no-cache-dir

echo %green%nvdiffrast build finished...%reset%

echo.
echo %green%Verifying nvdiffrast...%reset%
%PYTHON_PATH% -c "import nvdiffrast.torch as dr; print('nvdiffrast.torch OK')" || (
    echo %red%ERROR: nvdiffrast failed to load%reset%
    exit /b 1
)

:: ===============================
:: FlexGEMM
:: ===============================
echo.
echo %green%Building FlexGEMM...%reset%

%PYTHON_PATH% -m pip uninstall -y flex_gemm
%PYTHON_PATH% -m pip install git+https://github.com/JeffreyXiang/FlexGEMM --no-build-isolation --no-cache-dir

echo %green%FlexGEMM build finished...%reset%

:: ===============================
:: CuMesh
:: ===============================
echo.
echo %green%Building CuMesh...%reset%

%PYTHON_PATH% -m pip uninstall -y cumesh
%PYTHON_PATH% -m pip install git+https://github.com/JeffreyXiang/CuMesh --no-build-isolation --no-cache-dir

echo %green%CuMesh build finished...%reset%

:: ===============================
:: o-voxel
:: ===============================
echo.
echo %green%Building o_voxel...%reset%

pushd o-voxel
git submodule update --init --recursive
%PYTHON_PATH% -m pip uninstall -y o_voxel
%PYTHON_PATH% setup.py install
popd

echo %green%voxel build finished...%reset%

echo.
echo %green%Verifying voxel...%reset%
%PYTHON_PATH% -c "import o_voxel; print('Voxel OK')" || (
    echo %red%ERROR: Voxel failed to load%reset%
    exit /b 1
)

:: =================================================
:: Verification (core environment)
:: =================================================
echo.
echo %green%Verifying core environment...%reset%

%PYTHON_PATH% -c "import torch,numpy; print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('NumPy:',numpy.__version__)" || (
    echo %red%ERROR: Core env failed%reset%
    exit /b 1
)

:: =================================================
:: Verification (native extensions)
:: =================================================
echo.
echo %green%Verifying native extensions (best-effort)...%reset%

set "VERIFY_PY=%TEMP%\verify_native_ext.py"

echo import importlib > "%VERIFY_PY%"
echo mods = [^('cumesh','CuMesh'^),^('flex_gemm','FlexGEMM'^),^('o_voxel','Voxel'^),^('nvdiffrast','nvdiffrast'^)] >> "%VERIFY_PY%"
echo for m,n in mods: >> "%VERIFY_PY%"
echo.    try: >> "%VERIFY_PY%"
echo.        importlib.import_module(m) >> "%VERIFY_PY%"
echo.        print('[OK]',n) >> "%VERIFY_PY%"
echo.    except Exception as e: >> "%VERIFY_PY%"
echo.        print('[WARN]',n,'FAILED',e) >> "%VERIFY_PY%"

%PYTHON_PATH% "%VERIFY_PY%"
del "%VERIFY_PY%"

echo.
echo %green%=====================================%reset%
echo %green%   TRELLIS.2 INSTALL COMPLETE         %reset%
echo %green%=====================================%reset%
pause
exit /b 0

:: -------------------------------------------------
:: Helpers
:: -------------------------------------------------
:set_colors
set red=[91m
set green=[92m
set yellow=[93m
set reset=[0m
goto :eof

:get_versions
echo Checking versions...
echo.

for /f "tokens=2" %%i in ('"%PYTHON_PATH%" --version 2^>^&1') do (
    for /f "tokens=1,2 delims=." %%a in ("%%i") do set PYTHON_VERSION=%%a.%%b
)

"%PYTHON_PATH%" -c "import torch; print(torch.__version__)" > temp_torch.txt
for /f "tokens=1,2 delims=." %%a in (temp_torch.txt) do set TORCH_VERSION=%%a.%%b
del temp_torch.txt

"%PYTHON_PATH%" -c "import torch; print(torch.version.cuda)" > temp_cuda.txt
for /f "tokens=1,2 delims=." %%a in (temp_cuda.txt) do set CUDA_VERSION=%%a.%%b
del temp_cuda.txt

echo Python: %PYTHON_VERSION%
echo Torch : %TORCH_VERSION%
echo CUDA  : %CUDA_VERSION%
echo.
goto :eof
