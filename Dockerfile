# Use the official NVIDIA CUDA image as a base
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    wget \
    libjpeg-dev \
    libwebp-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Update pip and setuptools for correct installation of custom packages(nvdiffrast / utils3d etc.)
RUN pip install pip -U
RUN pip install setuptools -U

# Install basic Python dependencies
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard
RUN pip install pillow-simd

RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
RUN pip install kornia timm

# Copy assets and install flash-attention
RUN pip install flash_attn==2.7.3

# Install nvdiffrast
RUN mkdir -p /app/tmp/extensions
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /app/tmp/extensions/nvdiffrast
RUN pip install /app/tmp/extensions/nvdiffrast --no-build-isolation

# Install nvdiffrec
RUN mkdir -p /app/tmp/extensions
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /app/tmp/extensions/nvdiffrec
RUN pip install /app/tmp/extensions/nvdiffrec --no-build-isolation

# Install CuMesh
RUN mkdir -p /app/tmp/extensions
RUN git clone https://github.com/FishWoWater/CuMesh.git /app/tmp/extensions/CuMesh --recursive
RUN pip install /app/tmp/extensions/CuMesh --no-build-isolation

# Install FlexGEMM
RUN mkdir -p /app/tmp/extensions
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /app/tmp/extensions/FlexGEMM --recursive
RUN pip install /app/tmp/extensions/FlexGEMM --no-build-isolation

# Install o-voxel
COPY o-voxel /app/o-voxel
RUN mkdir -p /app/tmp/extensions
RUN cp -r /app/o-voxel /app/tmp/extensions/o-voxel
RUN pip install /app/tmp/extensions/o-voxel --no-build-isolation --no-deps

# Copy the rest of the application code
COPY . .

# Set the default command to run the application
# CMD ["python", "app.py"]
