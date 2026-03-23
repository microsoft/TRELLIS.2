@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: setup.bat  --  Windows equivalent of setup.sh for TRELLIS.2
:: ============================================================

set NEW_ENV=0
set BASIC=0
set FLASHATTN=0
set CUMESH=0
set OVOXEL=0
set FLEXGEMM=0
set NVDIFFRAST=0
set NVDIFFREC=0
set HELP=0

if "%~1"=="" set HELP=1

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="-h"           set HELP=1      & shift & goto parse_args
if /i "%~1"=="--help"       set HELP=1      & shift & goto parse_args
if /i "%~1"=="--new-env"    set NEW_ENV=1   & shift & goto parse_args
if /i "%~1"=="--basic"      set BASIC=1     & shift & goto parse_args
if /i "%~1"=="--flash-attn" set FLASHATTN=1 & shift & goto parse_args
if /i "%~1"=="--cumesh"     set CUMESH=1    & shift & goto parse_args
if /i "%~1"=="--o-voxel"    set OVOXEL=1    & shift & goto parse_args
if /i "%~1"=="--flexgemm"   set FLEXGEMM=1  & shift & goto parse_args
if /i "%~1"=="--nvdiffrast" set NVDIFFRAST=1 & shift & goto parse_args
if /i "%~1"=="--nvdiffrec"  set NVDIFFREC=1 & shift & goto parse_args
if /i "%~1"=="--texture"    set BASIC=1 & set FLASHATTN=1 & set OVOXEL=1 & set CUMESH=1 & set FLEXGEMM=1 & set NVDIFFRAST=1 & shift & goto parse_args
if /i "%~1"=="--all"        set NEW_ENV=1 & set BASIC=1 & set FLASHATTN=1 & set CUMESH=1 & set OVOXEL=1 & set FLEXGEMM=1 & set NVDIFFRAST=1 & set NVDIFFREC=1 & shift & goto parse_args
echo Error: Invalid argument: %~1
set HELP=1
shift & goto parse_args

:done_args

if %HELP%==1 (
    echo Usage: setup.bat [OPTIONS]
    echo Options:
    echo   -h, --help              Display this help message
    echo   --new-env               Create a new venv at .venv ^(requires Python 3.10^)
    echo   --basic                 Install basic dependencies
    echo   --flash-attn            Install flash-attention ^(CUDA only^)
    echo   --cumesh                Install cumesh
    echo   --o-voxel               Install o-voxel
    echo   --flexgemm              Install flexgemm
    echo   --nvdiffrast            Install nvdiffrast ^(CUDA only^)
    echo   --nvdiffrec             Install nvdiffrec ^(CUDA only^)
    echo   --texture               Install all dependencies for texturing ^(basic, flash-attn, o-voxel, cumesh, flexgemm, nvdiffrast^)
    echo   --all                   Run all of the above
    goto :eof
)

:: ---- Detect GPU platform ----------------------------------------
set PLATFORM=
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set PLATFORM=cuda
) else (
    rocminfo >nul 2>&1
    if !errorlevel!==0 (
        set PLATFORM=hip
    )
)

if "%PLATFORM%"=="" (
    echo Error: No supported GPU found ^(nvidia-smi and rocminfo both failed^)
    exit /b 1
)
echo Detected platform: %PLATFORM%

set WORKDIR=%CD%
set EXT_DIR=%TEMP%\trellis2_extensions

:: ---- Activate MSVC environment (needed for CUDA extension builds) ----
set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if exist %VCVARS% (
    call %VCVARS% x64
) else (
    echo Warning: vcvarsall.bat not found at %VCVARS%
    echo          CUDA extension builds ^(cumesh, flexgemm, nvdiffrast, nvdiffrec, o-voxel^) will likely fail.
    echo          Open a Developer Command Prompt for VS 2022 and re-run setup.bat if needed.
)

:: ---- --new-env --------------------------------------------------
if %NEW_ENV%==1 (
    python -m venv "%WORKDIR%\.venv"
    call "%WORKDIR%\.venv\Scripts\activate.bat"
    if "%PLATFORM%"=="cuda" (
        pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --no-build-isolation
    ) else if "%PLATFORM%"=="hip" (
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4 --no-build-isolation
    )
)

:: ---- --basic ----------------------------------------------------
if %BASIC%==1 (
    pip install packaging wheel setuptools imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard --no-build-isolation
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 --no-build-isolation
    :: NOTE: pillow-simd has no Windows wheel; install plain pillow instead
    pip install pillow --no-build-isolation
    pip install kornia timm --no-build-isolation
)

:: ---- --flash-attn -----------------------------------------------
if %FLASHATTN%==1 (
    if "%PLATFORM%"=="cuda" (
        pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2+cu128torch2.7-cp312-cp312-win_amd64.whl --no-build-isolation
    ) else (
        echo [FLASHATTN] Unsupported platform on Windows: %PLATFORM%
    )
)

:: ---- --nvdiffrast -----------------------------------------------
if %NVDIFFRAST%==1 (
    if "%PLATFORM%"=="cuda" (
        mkdir "%EXT_DIR%" 2>nul
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "%EXT_DIR%\nvdiffrast"
        pip install "%EXT_DIR%\nvdiffrast" --no-build-isolation
    ) else (
        echo [NVDIFFRAST] Unsupported platform: %PLATFORM%
    )
)

:: ---- --nvdiffrec ------------------------------------------------
if %NVDIFFREC%==1 (
    if "%PLATFORM%"=="cuda" (
        mkdir "%EXT_DIR%" 2>nul
        git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "%EXT_DIR%\nvdiffrec"
        pip install "%EXT_DIR%\nvdiffrec" --no-build-isolation
    ) else (
        echo [NVDIFFREC] Unsupported platform: %PLATFORM%
    )
)

:: ---- --cumesh ---------------------------------------------------
if %CUMESH%==1 (
    mkdir "%EXT_DIR%" 2>nul
    git clone https://github.com/JeffreyXiang/CuMesh.git "%EXT_DIR%\CuMesh" --recursive
    pip install "%EXT_DIR%\CuMesh" --no-build-isolation
)

:: ---- --flexgemm -------------------------------------------------
if %FLEXGEMM%==1 (
    mkdir "%EXT_DIR%" 2>nul
    git clone https://github.com/JeffreyXiang/FlexGEMM.git "%EXT_DIR%\FlexGEMM" --recursive
    pip install "%EXT_DIR%\FlexGEMM" --no-build-isolation
)

:: ---- --o-voxel --------------------------------------------------
if %OVOXEL%==1 (
    mkdir "%EXT_DIR%" 2>nul
    xcopy /E /I /Y "%WORKDIR%\o-voxel" "%EXT_DIR%\o-voxel"
    pip install "%EXT_DIR%\o-voxel" --no-build-isolation
)

echo.
echo Setup complete.
endlocal
