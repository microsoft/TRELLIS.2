#!/bin/bash

die() { # exit on error
    echo "ERROR: $1" >&2; exit 1
}

# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false


if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --nvdiffrec) NVDIFFREC=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --flash-attn            Install flash-attention"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --nvdiffrec             Install nvdiffrec"
    return
fi

# Get system information
WORKDIR=$(pwd)
if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    die "Error: No supported GPU found"
fi

# Source conda if installed
try_activate_conda() {
    # 1. is conda in $PATH?, 2. is it installed?, 3. source conda
    local CONDA_SEARCH_PATHS=("/opt/conda" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/conda")
    local BASE=""
    if command -v conda >/dev/null 2>&1; then
        BASE=$(conda info --base 2>/dev/null)
    else
        for p in "${CONDA_SEARCH_PATHS[@]}"; do
            [[ -f "$p/etc/profile.d/mamba.sh" || -f "$p/etc/profile.d/conda.sh" ]] && { BASE="$p"; break; }
        done
    fi
    [[ -z "$BASE" ]] && return 1 # conda not found

    # source both conda and mamba if available
    [[ -f "$BASE/etc/profile.d/conda.sh" ]] && source "$BASE/etc/profile.d/conda.sh"
    [[ -f "$BASE/etc/profile.d/mamba.sh" ]] && source "$BASE/etc/profile.d/mamba.sh"
    
    CURRENT="${CONDA_DEFAULT_ENV:-$(conda info --envs 2>/dev/null | grep '\*' | awk '{print $1}' | head -1)}" && CURRENT="${CURRENT:-none}"
    [[ -z ${CURRENT} || ${CURRENT} == "base" ]] && return 2 # do not install in conda base

    return 0
}

WITH_CONDA=""
try_activate_conda
case $? in
    0) WITH_CONDA=true ;;
    1) [[ "$NEW_ENV" = true ]] && die "No Conda found, cannot create environment" ;;
    2) [[ ! "$NEW_ENV" = true ]] && die "Conda found, current env: base, pass --new-env or switch to named env" ;;
esac

# prefer mamba, default to conda
conda() { command -v mamba >/dev/null 2>&1 && { mamba "$@"; return; }; command conda "$@"; }

if [ "$NEW_ENV" = true ] ; then
    WITH_CONDA=true
    CONDA_ENV="trellis2"
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
    CURRENT="${CONDA_DEFAULT_ENV:-$(conda info --envs 2>/dev/null | grep '\*' | awk '{print $1}' | head -1)}" && CURRENT="${CURRENT:-none}"
    [[ "$CURRENT" != "$CONDA_ENV" ]] && die "Error: Current conda environment $CURRENT is not $CONDA_ENV"
fi

# Get CUDA information
if [ "$PLATFORM" = "cuda" ] ; then  # nvdiffrast, cumesh, flexgmm, ovoxel require compiling on torch cuda version
    CUDA_EXPECT=12.4    # adjust to match torch
    CUDA_VERSION=$(nvcc --version -v | grep "release" | awk '{print $5}' | cut -d',' -f1)
    if [ "$CUDA_VERSION" != "$CUDA_EXPECT" ]; then
        if [ $WITH_CONDA ] ; then
            # if incompatible cuda version, but using conda, installs on environment and symlinks paths, reference solution from: 
            # https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer/blob/main/cosmos-predict1.yaml
            # https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer/blob/main/Dockerfile
            echo "Installing CUDA version ${CUDA_EXPECT} on conda environemnt ${CONDA_ENV}. Machine has version ${CUDA_VERSION}."
            conda install -y cmake gcc=${CUDA_EXPECT} gxx=${CUDA_EXPECT} cuda=${CUDA_EXPECT} cuda-nvcc=${CUDA_EXPECT} cuda-toolkit=${CUDA_EXPECT}
            CUDA_HOME=${CONDA_PREFIX}
            ln -sf ${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/*/include/* ${CONDA_PREFIX}/include/
            ln -sf ${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/*/include/* ${CONDA_PREFIX}/include/python3.10
            ln -sf ${CONDA_PREFIX}/lib/python3.10/site-packages/triton/backends/nvidia/include/* ${CONDA_PREFIX}/include/
        else
            die "Cuda version found, $CUDA_VERSION != expected version, $CUDA_EXPECT. Reinstall or use conda." 
        fi
    fi
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
elif [ "$PLATFORM" = "hip" ] ; then
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
fi

install_pillow_simd() {
    # avoid multiple pillow installs and crosslinked libraries
    if [ $WITH_CONDA ]; then
        conda uninstall -y --force jpeg libtiff
        conda install -y -c conda-forge libjpeg-turbo zlib gxx_linux-64 --no-deps
    elif [[ -z $(dpkg -l | grep libjpeg-dev) ]]; then
        echo "Installing libjpeg-dev, may require sudo password"
        sudo apt install -y libjpeg-dev
    fi
    pip install --upgrade pip setuptools wheel
    pip uninstall -y pillow

    CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
    [[ $WITH_CONDA ]] && conda install -y jpeg libtiff
}

if [ "$BASIC" = true ] ; then
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers==4.57.6 gradio==6.0.1 tensorboard pandas lpips zstandard
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    install_pillow_simd
    pip install kornia timm
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        # bypass flash-attn error https://github.com/Dao-AILab/flash-attention/issues/246 No Module Named 'torch'
        pip install psutil
        pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        pip install /tmp/extensions/nvdiffrast --no-build-isolation
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFREC" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
        pip install /tmp/extensions/nvdiffrec --no-build-isolation
    else
        echo "[NVDIFFREC] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$CUMESH" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    pip install /tmp/extensions/CuMesh --no-build-isolation
fi

if [ "$FLEXGEMM" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    pip install /tmp/extensions/FlexGEMM --no-build-isolation
fi

if [ "$OVOXEL" = true ] ; then
    mkdir -p /tmp/extensions
    cp -r o-voxel /tmp/extensions/o-voxel
    pip install /tmp/extensions/o-voxel --no-build-isolation
fi
