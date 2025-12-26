"""
Modal image definition for TRELLIS.2.

This module defines the container image with all dependencies pre-installed,
including CUDA extensions that are compiled at image build time rather than
runtime. This ensures reproducible builds and eliminates cold start compilation.

Build notes:
- gpu="T4" in run_commands creates a FRESH build context
- ALL dependencies (torch, wheel, setuptools) must be in the GPU block
- --no-build-isolation needed for packages requiring torch at build time
- clang required for CUDA extension builds
- CUDA 12.4 required for TRELLIS.2
"""

import modal

# Constants duplicated from config.py - Modal copies this file to /root/image.py
# without the trellis2_modal package, so imports fail. Keep in sync manually.
# Verified by tests/test_config_consistency.py
GPU_TYPE = "A100-80GB"
HF_CACHE_PATH = "/cache/huggingface"
MODEL_NAME = "microsoft/TRELLIS.2-4B"

# Pinned git commits for reproducible builds
# These commits are validated to work with PyTorch 2.6.0 / CUDA 12.4
PINNED_COMMITS = {
    "nvdiffrast": "253ac4fcea7de5f396371124af597e6cc957bfae",  # v0.4.0 tag
    "nvdiffrec": "b296927cc7fd01c2ac1087c8065c4d7248f72da4",  # renderutils branch
    "utils3d": "9a4eb15e4021b67b12c460c7057d642626897ec8",  # TRELLIS integration
    "cumesh": "d8d28794721a3f4984b1b12c24403f546f41d28c",  # HEAD 2025-12-20
    "flex_gemm": "8b9afa2d56f667b709ccd761d0bd7aab48bdd7cf",  # HEAD 2025-12-20
    "trellis2": "1762f493fe7731a3b7cc6b79ad5da7b015b516c1",  # HEAD 2025-12-20
}

# Build paths (avoid magic strings)
BUILD_TMP = "/tmp"
TRELLIS2_PATH = "/opt/TRELLIS.2"

# Modal app for TRELLIS.2
app = modal.App("trellis2-3d")

# Modal volumes for persistent storage
hf_cache_volume = modal.Volume.from_name("trellis2-hf-cache", create_if_missing=True)
api_keys_volume = modal.Volume.from_name("trellis2-api-keys", create_if_missing=True)

# HuggingFace secret for accessing gated models (dinov3, etc.)
# Create with: modal secret create huggingface HF_TOKEN=hf_xxxxx
hf_secret = modal.Secret.from_name("huggingface")

# System packages required for CUDA extensions and TRELLIS.2 dependencies
SYSTEM_PACKAGES = [
    "git",
    "ninja-build",
    "cmake",
    "build-essential",
    "clang",  # Required for CUDA extension builds
    "libgl1-mesa-glx",  # OpenGL for nvdiffrast
    "libglib2.0-0",
    "libjpeg-dev",
    "libpng-dev",
    "libgomp1",  # OpenMP for parallel processing
    "libopenexr-dev",  # For HDRI loading with OpenCV
]

# Core Python dependencies (no GPU needed for install)
CORE_PYTHON_PACKAGES = [
    "pillow",
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "easydict",
    "opencv-python-headless",
    "scipy",
    "ninja",
    "trimesh",
    "transformers",
    "huggingface-hub",
    "safetensors",
    "lz4",  # State compression
    "kornia",  # Image processing
    "timm",  # Vision models
]

# TRELLIS.2 image with all CUDA extensions pre-compiled
trellis2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(*SYSTEM_PACKAGES)
    .pip_install(*CORE_PYTHON_PACKAGES)
    # GPU build block - ALL CUDA-related builds in ONE block
    # because gpu="T4" creates a fresh build context
    .run_commands(
        # Install PyTorch with CUDA 12.4
        "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
        # Build tools for --no-build-isolation
        "pip install wheel setuptools",
        # flash-attn build dependencies
        "pip install psutil packaging",
        # flash-attn (needs torch at build time, use --no-build-isolation)
        "pip install flash-attn==2.7.3 --no-build-isolation",
        # utils3d (pinned version for TRELLIS.2)
        f"pip install git+https://github.com/EasternJournalist/utils3d.git@{PINNED_COMMITS['utils3d']}",
        # nvdiffrast (builds from source, pinned)
        f"git clone https://github.com/NVlabs/nvdiffrast.git {BUILD_TMP}/nvdiffrast && cd {BUILD_TMP}/nvdiffrast && git checkout {PINNED_COMMITS['nvdiffrast']}",
        f"pip install {BUILD_TMP}/nvdiffrast --no-build-isolation",
        # nvdiffrec renderutils (needs torch at build time, pinned)
        f"git clone https://github.com/JeffreyXiang/nvdiffrec.git {BUILD_TMP}/nvdiffrec && cd {BUILD_TMP}/nvdiffrec && git checkout {PINNED_COMMITS['nvdiffrec']}",
        f"pip install {BUILD_TMP}/nvdiffrec --no-build-isolation",
        # CuMesh (needs torch at build time, pinned)
        f"git clone --recursive https://github.com/JeffreyXiang/CuMesh.git {BUILD_TMP}/CuMesh && cd {BUILD_TMP}/CuMesh && git checkout {PINNED_COMMITS['cumesh']}",
        f"pip install {BUILD_TMP}/CuMesh --no-build-isolation",
        # FlexGEMM (needs torch/triton at build time, pinned)
        f"git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git {BUILD_TMP}/FlexGEMM && cd {BUILD_TMP}/FlexGEMM && git checkout {PINNED_COMMITS['flex_gemm']}",
        f"pip install {BUILD_TMP}/FlexGEMM --no-build-isolation",
        gpu="T4",
    )
    # Clone TRELLIS.2 repository (with submodules for o-voxel)
    .run_commands(
        f"git clone --recursive https://github.com/microsoft/TRELLIS.2.git {TRELLIS2_PATH} && cd {TRELLIS2_PATH} && git checkout {PINNED_COMMITS['trellis2']}",
    )
    # Build o-voxel from the repo's submodule
    .run_commands(
        f"pip install {TRELLIS2_PATH}/o-voxel --no-build-isolation",
        gpu="T4",
    )
    # Pre-download DINOv2 model to avoid runtime download
    .run_commands(
        "python -c \"import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)\"",
        gpu="T4",
    )
    # Set environment variables
    .env(
        {
            "ATTN_BACKEND": "flash_attn",
            "PYTHONPATH": TRELLIS2_PATH,
            "HF_HOME": HF_CACHE_PATH,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        }
    )
)


@app.function(image=trellis2_image, gpu="T4", timeout=600)
def verify_image():
    """
    Verify the image is correctly built with all dependencies.

    Run with: modal run trellis2_modal/service/image.py::verify_image
    """
    import json

    results = {}

    # Test PyTorch + CUDA
    import torch

    results["pytorch_version"] = str(torch.__version__)
    results["cuda_version"] = str(torch.version.cuda)
    results["cuda_available"] = torch.cuda.is_available()
    results["device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )

    # Test CUDA compute
    if torch.cuda.is_available():
        x = torch.randn(100, 100, device="cuda")
        _ = torch.matmul(x, x)
        results["cuda_compute_works"] = True

    # Test flash-attn
    try:
        from flash_attn import flash_attn_func

        results["flash_attn"] = callable(flash_attn_func)
    except ImportError as e:
        results["flash_attn"] = str(e)

    # Test CUDA extensions
    try:
        import nvdiffrast.torch as dr

        results["nvdiffrast"] = hasattr(dr, "rasterize")
    except ImportError as e:
        results["nvdiffrast"] = str(e)

    try:
        import cumesh

        results["cumesh"] = hasattr(cumesh, "CuMesh")
    except ImportError as e:
        results["cumesh"] = str(e)

    try:
        import flex_gemm  # noqa: F401

        results["flex_gemm"] = True
    except ImportError as e:
        results["flex_gemm"] = str(e)

    # Test o_voxel
    try:
        from o_voxel.postprocess import to_glb

        results["o_voxel"] = callable(to_glb)
    except ImportError as e:
        results["o_voxel"] = str(e)

    # Test utils3d
    try:
        import utils3d  # noqa: F401

        results["utils3d"] = True
    except ImportError as e:
        results["utils3d"] = str(e)

    # Test DINOv2 is cached
    import os

    dinov2_hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    results["dinov2_cached"] = os.path.exists(dinov2_hub)

    # Test TRELLIS.2 imports
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401

        results["trellis2_pipeline"] = True
    except ImportError as e:
        results["trellis2_pipeline"] = str(e)

    try:
        from trellis2.utils import render_utils

        results["trellis2_render_utils"] = hasattr(render_utils, "render_video")
    except ImportError as e:
        results["trellis2_render_utils"] = str(e)

    # Test HDRI file exists (TRELLIS2_PATH is /opt/TRELLIS.2)
    hdri_path = f"{TRELLIS2_PATH}/assets/hdri/forest.exr"
    results["hdri_exists"] = os.path.exists(hdri_path)

    print("=== Image Verification Results ===")
    for k, v in sorted(results.items()):
        status = "✓" if v is True else "✗" if v is False else "?"
        print(f"  {status} {k}: {v}")

    return json.dumps(results)


@app.local_entrypoint()
def main():
    """Run verification."""
    import json

    print("\n" + "=" * 60)
    print("TRELLIS.2 Modal Image Verification")
    print("=" * 60 + "\n")

    result = verify_image.remote()
    data = json.loads(result)

    # Check critical components
    critical = [
        "cuda_available",
        "flash_attn",
        "nvdiffrast",
        "cumesh",
        "flex_gemm",
        "o_voxel",
        "trellis2_pipeline",
        "trellis2_render_utils",
        "dinov2_cached",
        "hdri_exists",
    ]

    all_passed = all(data.get(k, False) is True for k in critical)

    if all_passed:
        print("\n✓ Image verification PASSED")
    else:
        print("\n✗ Image verification FAILED")
        for k in critical:
            v = data.get(k, "MISSING")
            status = "✓" if v is True else "✗"
            print(f"  {status} {k}: {v}")
