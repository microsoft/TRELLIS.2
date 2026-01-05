# syntax=docker/dockerfile:1-labs
# docker buildx create --name gpubuilder --driver-opt "image=moby/buildkit:buildx-stable-1-gpu" --bootstrap
# docker buildx --builder gpubuilder build --allow device -t trellis --load .
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y g++ git cmake ninja-build \
    python3 python-is-python3 python3-pip python3-venv sudo
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
# create venv instead of conde (see --new-env in setup.sh)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -U wheel setuptools pip
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
WORKDIR /TRELLIS.2
ARG TORCH_CUDA_ARCH_LIST="8.0"
RUN --device=nvidia.com/gpu bash -c ". ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]