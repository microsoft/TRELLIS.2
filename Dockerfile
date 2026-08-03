# CUDA 12.4 matches the PyTorch wheel used by the upstream setup script.
# The host only needs a compatible NVIDIA driver plus NVIDIA Container Toolkit.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST=8.9
ARG MAX_JOBS=4

ENV TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface/transformers \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        gcc-11 \
        g++-11 \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libjpeg-dev \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110 \
    && python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# The dependency layer needs only o-voxel. Keeping application source out of
# this layer lets Docker reuse the expensive CUDA-extension build when app code
# or the Compose configuration changes.
COPY o-voxel /workspace/o-voxel

# Keep these commands in the same order as setup.sh, excluding conda and sudo.
RUN python3 -m pip install \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless==4.10.0.84 Pillow==9.5.0 ninja trimesh \
        transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
        kornia timm \
    && python3 -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 \
    && python3 -m pip install flash-attn==2.7.3 --no-build-isolation \
    && git clone --depth 1 --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
    && python3 -m pip install /tmp/nvdiffrast --no-build-isolation \
    && git clone --depth 1 --branch renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec \
    && python3 -m pip install /tmp/nvdiffrec --no-build-isolation \
    && git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/CuMesh.git /tmp/cumesh \
    && python3 -m pip install /tmp/cumesh --no-build-isolation \
    && git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/FlexGEMM.git /tmp/flexgemm \
    && python3 -m pip install /tmp/flexgemm --no-build-isolation \
    && python3 -m pip install ./o-voxel --no-build-isolation \
    && rm -rf /tmp/nvdiffrast /tmp/nvdiffrec /tmp/cumesh /tmp/flexgemm

COPY . /workspace

RUN mkdir -p /models/huggingface /workspace/outputs

EXPOSE 7860

CMD ["python3", "app.py"]
