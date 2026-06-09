"""
Configuration constants for the Modal TRELLIS.2 service.

Contains deployment settings, resource limits, and path configurations
that are shared across the service modules.

NOTE: Some constants are duplicated in image.py because Modal copies
that file in isolation during image builds (imports fail). The test
test_config_consistency.py verifies these stay in sync.
"""

# Modal resource configuration
GPU_TYPE = "A100-80GB"
CONTAINER_IDLE_TIMEOUT = 300  # seconds

# GPU snapshots don't help: flex_gemm/Triton reinit negates benefits (~143s vs ~146s)
GPU_MEMORY_SNAPSHOT = False

# Volume mount paths
HF_CACHE_PATH = "/cache/huggingface"
API_KEYS_PATH = "/data/keys.json"

# Build/runtime paths
TRELLIS2_PATH = "/opt/TRELLIS.2"

# Model configuration
MODEL_NAME = "microsoft/TRELLIS.2-4B"

# Input validation limits
MAX_IMAGE_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10MB (decoded binary size)
MAX_IMAGE_DIMENSION = 4096  # Max width or height in pixels

# Generation defaults (from pipeline.json, can be overridden per-request)
DEFAULT_SEED = 42
DEFAULT_PIPELINE_TYPE = "1024_cascade"

# Sparse structure sampler defaults
DEFAULT_SS_SAMPLING_STEPS = 12
DEFAULT_SS_GUIDANCE_STRENGTH = 7.5

# Shape SLAT sampler defaults
DEFAULT_SHAPE_SLAT_SAMPLING_STEPS = 12
DEFAULT_SHAPE_SLAT_GUIDANCE_STRENGTH = 7.5

# Texture SLAT sampler defaults
DEFAULT_TEX_SLAT_SAMPLING_STEPS = 12
DEFAULT_TEX_SLAT_GUIDANCE_STRENGTH = 1.0

# GLB extraction defaults (from official example.py)
DEFAULT_DECIMATION_TARGET = 1_000_000
DEFAULT_TEXTURE_SIZE = 4096
DEFAULT_REMESH = True
DEFAULT_REMESH_BAND = 1.0
DEFAULT_REMESH_PROJECT = 0.0
