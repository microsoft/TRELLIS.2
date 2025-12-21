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
    echo "Error: No supported GPU found"
    exit 1
fi

if [ "$NEW_ENV" = true ] ; then
    conda create -n trellis2 python=3.10
    conda activate trellis2
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    elif [ "$PLATFORM" = "hip" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    fi
fi

if [ "$BASIC" = true ] ; then
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    sudo apt install -y libjpeg-dev
    pip install pillow-simd
    pip install kornia timm
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        # flash-attn's setup.py imports torch during build requirements phase
        # pip's isolated build environment doesn't have torch, so we build from source
        echo "[FLASHATTN] Building from source (torch required during build)..."
        mkdir -p /tmp/extensions
        if [ ! -d "/tmp/extensions/flash-attention" ]; then
            git clone --recursive https://github.com/Dao-AILab/flash-attention.git /tmp/extensions/flash-attention
        fi
        cd /tmp/extensions/flash-attention
        git fetch --tags
        git checkout v2.7.3 2>/dev/null || git checkout tags/v2.7.3 2>/dev/null || git checkout v2.7.3
        git submodule update --init --recursive
        # Build with pip install to ensure torch is available in the environment
        MAX_JOBS=4 pip install . --no-build-isolation
        cd $WORKDIR
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
        if [ ! -d "/tmp/extensions/nvdiffrec" ]; then
            git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
        fi
        
        # Find libcuda.so (NVIDIA driver library)
        LIBCUDA_PATH=$(find /usr -name "libcuda.so*" 2>/dev/null | head -1)
        if [ -z "$LIBCUDA_PATH" ]; then
            # Try wider search
            LIBCUDA_PATH=$(find / -name "libcuda.so*" 2>/dev/null | grep -v "/proc\|/sys\|/dev" | head -1)
        fi
        
        if [ -n "$LIBCUDA_PATH" ]; then
            LIBCUDA_DIR=$(dirname "$LIBCUDA_PATH")
            echo "[NVDIFFREC] Found libcuda.so at: $LIBCUDA_PATH"
            
            # Set library paths for the linker
            export LIBRARY_PATH=${LIBRARY_PATH}:${LIBCUDA_DIR}
            export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIBCUDA_DIR}
            
            # Also add common CUDA library locations
            if [ -n "$CUDA_HOME" ]; then
                export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
            else
                export LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/cuda-12.4/lib64:/usr/local/cuda/lib64
            fi
            
            # Set LDFLAGS to help the linker find libcuda
            export LDFLAGS="-L${LIBCUDA_DIR} ${LDFLAGS}"
        else
            echo "[NVDIFFREC] Warning: libcuda.so not found. Trying default locations..."
            export LIBRARY_PATH=${LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu:/usr/lib64
            export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu:/usr/lib64
        fi
        
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
