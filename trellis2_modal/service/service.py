"""
Modal service for TRELLIS.2 3D generation.

This module defines the Modal classes for 3D generation:
- HealthCheckService: CPU-only health endpoint (no GPU needed)
- TRELLIS2Service: GPU-accelerated generation endpoints

Usage:
    modal run -m trellis2_modal.service.service              # Test health check
    modal deploy -m trellis2_modal.service.service           # Deploy with snapshots

Endpoints:
    GET  /health       - Health check (no auth, CPU-only, no GPU allocation)
    POST /generate     - Image → 3D state + video (Modal Proxy Auth required)
    POST /extract_glb  - State → GLB mesh (Modal Proxy Auth required)

Authentication:
    POST endpoints use Modal Proxy Auth Tokens. Clients must provide:
    - Modal-Key: <your-key>
    - Modal-Secret: <your-secret>
    Create tokens in the Modal dashboard.

Scaling:
    max_containers=1 limits GPU container count to prevent quota exhaustion.
    Retry policy handles transient OOM failures with exponential backoff.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import modal
from fastapi import Request

from .auth import mask_api_key  # Only needed for logging
from .config import (
    API_KEYS_PATH,
    DEFAULT_DECIMATION_TARGET,
    DEFAULT_PIPELINE_TYPE,
    DEFAULT_REMESH,
    DEFAULT_REMESH_BAND,
    DEFAULT_REMESH_PROJECT,
    DEFAULT_SEED,
    DEFAULT_SHAPE_SLAT_GUIDANCE_STRENGTH,
    DEFAULT_SHAPE_SLAT_SAMPLING_STEPS,
    DEFAULT_SS_GUIDANCE_STRENGTH,
    DEFAULT_SS_SAMPLING_STEPS,
    DEFAULT_TEX_SLAT_GUIDANCE_STRENGTH,
    DEFAULT_TEX_SLAT_SAMPLING_STEPS,
    DEFAULT_TEXTURE_SIZE,
    GPU_MEMORY_SNAPSHOT,
    GPU_TYPE,
    HF_CACHE_PATH,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PAYLOAD_SIZE,
)
from .generator import TRELLIS2Generator
from .image import api_keys_volume, app, hf_cache_volume, hf_secret, trellis2_image

logger = logging.getLogger(__name__)

# Derive volume path from config (parent directory of API_KEYS_PATH)
# Note: os is imported at top of file via __future__
import os as _os  # noqa: E402

API_KEYS_VOLUME_PATH = _os.path.dirname(API_KEYS_PATH)

# Valid pipeline types
VALID_PIPELINE_TYPES = frozenset({"512", "1024", "1024_cascade", "1536_cascade"})

# Valid texture sizes
VALID_TEXTURE_SIZES = frozenset({512, 1024, 2048, 4096})

if TYPE_CHECKING:
    from PIL import Image


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return f"req_{secrets.token_hex(8)}"


@dataclass
class GenerateParams:
    """Validated parameters for the generate endpoint."""

    image: Image.Image
    seed: int
    pipeline_type: str
    ss_sampling_steps: int
    ss_guidance_strength: float
    shape_slat_sampling_steps: int
    shape_slat_guidance_strength: float
    tex_slat_sampling_steps: int
    tex_slat_guidance_strength: float


@dataclass
class ExtractGLBParams:
    """Validated parameters for the extract_glb endpoint."""

    state: dict[str, Any]
    decimation_target: int
    texture_size: int
    remesh: bool
    remesh_band: float
    remesh_project: float


def _error_response(code: str, message: str) -> dict:
    """Create a standardized error response."""
    return {"error": {"code": code, "message": message}}


def _log_request(
    endpoint: str,
    api_key: str | None,
    duration_ms: float,
    status: str,
    error_code: str | None = None,
    request_id: str | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    """Log structured request completion."""
    log_data: dict[str, Any] = {
        "endpoint": endpoint,
        "api_key": mask_api_key(api_key),
        "duration_ms": round(duration_ms, 2),
        "status": status,
    }
    if request_id:
        log_data["request_id"] = request_id
    if error_code:
        log_data["error_code"] = error_code
    if extra_metrics:
        log_data.update(extra_metrics)

    if status == "success":
        logger.info("Request completed: %s", log_data)
    else:
        logger.warning("Request failed: %s", log_data)


def _parse_generate_request(request: dict) -> GenerateParams | dict:
    """
    Parse and validate a generate request.

    Returns:
        GenerateParams if valid, or error dict if validation fails.
    """
    import base64
    from io import BytesIO

    from PIL import Image

    if not isinstance(request, dict):
        return _error_response("validation_error", "Request must be JSON object")

    # Required field
    image_b64 = request.get("image")
    if not image_b64:
        return _error_response("validation_error", "Missing 'image' field")

    # Check payload size
    estimated_size = len(image_b64) * 3 // 4
    if estimated_size > MAX_IMAGE_PAYLOAD_SIZE:
        return _error_response(
            "validation_error",
            f"Image size exceeds limit ({estimated_size // (1024 * 1024)}MB > "
            f"{MAX_IMAGE_PAYLOAD_SIZE // (1024 * 1024)}MB)",
        )

    # Parse parameters with defaults
    try:
        seed = int(request.get("seed", DEFAULT_SEED))
        pipeline_type = str(request.get("pipeline_type", DEFAULT_PIPELINE_TYPE))
        ss_sampling_steps = int(
            request.get("ss_sampling_steps", DEFAULT_SS_SAMPLING_STEPS)
        )
        ss_guidance_strength = float(
            request.get("ss_guidance_strength", DEFAULT_SS_GUIDANCE_STRENGTH)
        )
        shape_slat_sampling_steps = int(
            request.get("shape_slat_sampling_steps", DEFAULT_SHAPE_SLAT_SAMPLING_STEPS)
        )
        shape_slat_guidance_strength = float(
            request.get(
                "shape_slat_guidance_strength", DEFAULT_SHAPE_SLAT_GUIDANCE_STRENGTH
            )
        )
        tex_slat_sampling_steps = int(
            request.get("tex_slat_sampling_steps", DEFAULT_TEX_SLAT_SAMPLING_STEPS)
        )
        tex_slat_guidance_strength = float(
            request.get(
                "tex_slat_guidance_strength", DEFAULT_TEX_SLAT_GUIDANCE_STRENGTH
            )
        )
    except (TypeError, ValueError) as e:
        return _error_response("validation_error", f"Invalid parameter: {e}")

    # Validate pipeline_type
    if pipeline_type not in VALID_PIPELINE_TYPES:
        return _error_response(
            "validation_error",
            f"Invalid pipeline_type '{pipeline_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_PIPELINE_TYPES))}",
        )

    # Decode image
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return _error_response("validation_error", f"Invalid image: {e}")

    # Validate image dimensions
    width, height = image.size
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return _error_response(
            "validation_error",
            f"Image dimensions exceed limit "
            f"({width}x{height} > {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})",
        )

    return GenerateParams(
        image=image,
        seed=seed,
        pipeline_type=pipeline_type,
        ss_sampling_steps=ss_sampling_steps,
        ss_guidance_strength=ss_guidance_strength,
        shape_slat_sampling_steps=shape_slat_sampling_steps,
        shape_slat_guidance_strength=shape_slat_guidance_strength,
        tex_slat_sampling_steps=tex_slat_sampling_steps,
        tex_slat_guidance_strength=tex_slat_guidance_strength,
    )


def _parse_extract_glb_request(request: dict) -> ExtractGLBParams | dict:
    """
    Parse and validate an extract_glb request.

    Returns:
        ExtractGLBParams if valid, or error dict if validation fails.
    """
    import base64

    from trellis2_modal.client.compression import decompress_state

    if not isinstance(request, dict):
        return _error_response("validation_error", "Request must be JSON object")

    state_b64 = request.get("state")
    if not state_b64:
        return _error_response("validation_error", "Missing 'state' field")

    # Parse parameters with defaults
    try:
        decimation_target = int(
            request.get("decimation_target", DEFAULT_DECIMATION_TARGET)
        )
        texture_size = int(request.get("texture_size", DEFAULT_TEXTURE_SIZE))
        remesh = bool(request.get("remesh", DEFAULT_REMESH))
        remesh_band = float(request.get("remesh_band", DEFAULT_REMESH_BAND))
        remesh_project = float(request.get("remesh_project", DEFAULT_REMESH_PROJECT))
    except (TypeError, ValueError) as e:
        return _error_response("validation_error", f"Invalid parameter: {e}")

    # Validate ranges
    if decimation_target < 1000:
        return _error_response(
            "validation_error",
            "decimation_target must be at least 1000",
        )
    if texture_size not in VALID_TEXTURE_SIZES:
        return _error_response(
            "validation_error",
            f"texture_size must be one of: {', '.join(map(str, sorted(VALID_TEXTURE_SIZES)))}",
        )

    # Decode and decompress state
    try:
        compressed_state = base64.b64decode(state_b64)
        state = decompress_state(compressed_state)
    except Exception as e:
        return _error_response("validation_error", f"Invalid state: {e}")

    return ExtractGLBParams(
        state=state,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
    )


# CPU-only health check class - prevents health checks from spinning up GPU containers
# This uses a minimal image and no GPU, so health probes are fast and cheap
@app.cls(
    image=modal.Image.debian_slim().pip_install("fastapi[standard]"),
    cpu=1.0,
    min_containers=1,  # Keep one warm for instant health responses
)
class HealthCheckService:
    """
    Lightweight CPU-only health check service.

    Separated from GPU service to prevent load balancer health probes
    from consuming expensive GPU resources. This class can handle
    thousands of health checks per hour at minimal cost.
    """

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """GET /health - Health check for load balancers. No auth required."""
        return {"status": "ok", "service": "trellis2-api"}


@app.cls(
    image=trellis2_image,
    gpu=GPU_TYPE,
    secrets=[hf_secret],
    volumes={
        HF_CACHE_PATH: hf_cache_volume,
        API_KEYS_VOLUME_PATH: api_keys_volume,
    },
    timeout=600,
    scaledown_window=300,
    enable_memory_snapshot=GPU_MEMORY_SNAPSHOT,
    max_containers=1,  # Limit GPU container count to prevent quota exhaustion
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
)
class TRELLIS2Service:
    """
    Modal class wrapping TRELLIS2Generator.

    Uses composition to separate Modal infrastructure from domain logic.

    Note: CPU memory snapshots are not supported because TRELLIS.2 dependencies
    (flex_gemm, Triton) require GPU access during import. Model loading happens
    entirely in the GPU phase.
    """

    @modal.enter()
    def load_model(self) -> None:
        """Load model to GPU. Called once when container starts."""
        import torch

        self._load_start = time.time()
        self.generator = TRELLIS2Generator()

        logger.info("Loading model to GPU...")
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

        self.generator.load_model()

        logger.info("Model loaded: %.2fs", self.generator.load_time)
        logger.info("VRAM: %.2f GB", torch.cuda.memory_allocated() / 1e9)

    @modal.method()
    def health_check(self) -> dict:
        """Return health status and diagnostic information."""
        import torch

        return {
            "status": "healthy" if self.generator.is_ready else "unhealthy",
            "gpu": torch.cuda.get_device_name(0),
            "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "vram_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            ),
            "load_time_seconds": round(self.generator.load_time, 2),
        }

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate(self, http_request: Request, request: dict) -> dict:
        """
        POST /generate - Generate 3D model from image.

        Requires Modal Proxy Auth (Modal-Key/Modal-Secret headers).
        Create tokens in the Modal dashboard.

        Args:
            http_request: FastAPI Request for header access
            request: Dict with image (base64), seed, pipeline_type, sampler params

        Returns:
            Dict with state (compressed, base64), video (base64), request_id
        """
        import base64

        import torch

        from trellis2_modal.client.compression import compress_state

        request_id = generate_request_id()
        start_time = time.perf_counter()
        # Note: Modal Proxy Auth handles authentication at the proxy layer
        # Requests that reach this code are already authenticated
        api_key = http_request.headers.get("Modal-Key", "proxy-auth")

        # Parse request
        params = _parse_generate_request(request)
        if isinstance(params, dict):
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "generate",
                api_key,
                duration_ms,
                "error",
                "validation_error",
                request_id,
            )
            return params

        # Build sampler param dicts
        # Note: TRELLIS.2 samplers use "guidance_strength" (not "cfg_strength")
        ss_params = {
            "steps": params.ss_sampling_steps,
            "guidance_strength": params.ss_guidance_strength,
        }
        shape_params = {
            "steps": params.shape_slat_sampling_steps,
            "guidance_strength": params.shape_slat_guidance_strength,
        }
        tex_params = {
            "steps": params.tex_slat_sampling_steps,
            "guidance_strength": params.tex_slat_guidance_strength,
        }

        # Generate 3D
        # OOM handling pattern: catch outside exception scope to properly free memory
        # (Python exception frames hold tensor references, preventing empty_cache from working)
        gen_oom = False
        gen_error = None
        state = None
        try:
            state = self.generator.generate(
                image=params.image,
                seed=params.seed,
                pipeline_type=params.pipeline_type,
                ss_params=ss_params,
                shape_params=shape_params,
                tex_params=tex_params,
            )
        except torch.cuda.OutOfMemoryError:
            gen_oom = True
        except Exception as e:
            gen_error = e

        if gen_oom:
            torch.cuda.empty_cache()
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "generate", api_key, duration_ms, "error", "cuda_oom", request_id
            )
            return _error_response(
                "cuda_oom",
                "GPU out of memory. Try a smaller image or lower resolution pipeline.",
            )

        if gen_error is not None:
            logger.exception("Generation failed: %s", gen_error)
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "generate",
                api_key,
                duration_ms,
                "error",
                "generation_error",
                request_id,
            )
            return _error_response("generation_error", f"Generation failed: {gen_error}")

        # Render preview video
        # OOM handling pattern: catch outside exception scope
        video_oom = False
        video_error = None
        video_bytes = None
        try:
            video_bytes = self.generator.render_preview_video(state)
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        except torch.cuda.OutOfMemoryError:
            video_oom = True
        except Exception as e:
            video_error = e

        if video_oom:
            torch.cuda.empty_cache()
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "generate", api_key, duration_ms, "error", "cuda_oom", request_id
            )
            return _error_response(
                "cuda_oom",
                "GPU out of memory during video rendering.",
            )

        if video_error is not None:
            logger.exception("Video rendering failed: %s", video_error)
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "generate",
                api_key,
                duration_ms,
                "error",
                "rendering_error",
                request_id,
            )
            return _error_response("rendering_error", f"Video rendering failed: {video_error}")

        # Compress state
        try:
            compressed_state = compress_state(state)
            state_b64 = base64.b64encode(compressed_state).decode("utf-8")
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "generate",
                api_key,
                duration_ms,
                "error",
                "compression_error",
                request_id,
            )
            return _error_response(
                "compression_error", f"State compression failed: {e}"
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        _log_request(
            "generate",
            api_key,
            duration_ms,
            "success",
            request_id=request_id,
            extra_metrics={
                "state_size_bytes": len(compressed_state),
                "video_size_bytes": len(video_bytes),
            },
        )
        return {
            "state": state_b64,
            "video": video_b64,
            "request_id": request_id,
        }

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def extract_glb(self, http_request: Request, request: dict) -> dict:
        """
        POST /extract_glb - Extract GLB mesh from generation state.

        Requires Modal Proxy Auth (Modal-Key/Modal-Secret headers).
        Create tokens in the Modal dashboard.

        Args:
            http_request: FastAPI Request for header access
            request: Dict with state (base64), decimation params

        Returns:
            Dict with glb (base64), request_id
        """
        import base64

        import torch

        request_id = generate_request_id()
        start_time = time.perf_counter()
        # Note: Modal Proxy Auth handles authentication at the proxy layer
        # Requests that reach this code are already authenticated
        api_key = http_request.headers.get("Modal-Key", "proxy-auth")

        # Parse request
        params = _parse_extract_glb_request(request)
        if isinstance(params, dict):
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "extract_glb",
                api_key,
                duration_ms,
                "error",
                "validation_error",
                request_id,
            )
            return params

        # Extract GLB
        # OOM handling pattern: catch outside exception scope
        glb_oom = False
        glb_error = None
        glb_bytes = None
        try:
            glb_bytes = self.generator.extract_glb(
                state=params.state,
                decimation_target=params.decimation_target,
                texture_size=params.texture_size,
                remesh=params.remesh,
                remesh_band=params.remesh_band,
                remesh_project=params.remesh_project,
            )
        except torch.cuda.OutOfMemoryError:
            glb_oom = True
        except Exception as e:
            glb_error = e

        if glb_oom:
            torch.cuda.empty_cache()
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "extract_glb", api_key, duration_ms, "error", "cuda_oom", request_id
            )
            return _error_response(
                "cuda_oom",
                "GPU out of memory during GLB extraction. Try reducing texture size.",
            )

        if glb_error is not None:
            logger.exception("GLB extraction failed: %s", glb_error)
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request(
                "extract_glb",
                api_key,
                duration_ms,
                "error",
                "extraction_error",
                request_id,
            )
            return _error_response("extraction_error", f"GLB extraction failed: {glb_error}")

        glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")

        duration_ms = (time.perf_counter() - start_time) * 1000
        _log_request(
            "extract_glb",
            api_key,
            duration_ms,
            "success",
            request_id=request_id,
            extra_metrics={"glb_size_bytes": len(glb_bytes)},
        )
        return {"glb": glb_b64, "request_id": request_id}


@app.local_entrypoint()
def test_service():
    """Test the service with a health check."""
    import json

    print("\n" + "=" * 60)
    print("TRELLIS.2 Modal Service - Health Check")
    print("=" * 60 + "\n")

    service = TRELLIS2Service()

    print("Starting first call (cold start)...")
    start = time.time()
    result = service.health_check.remote()
    first_call_time = time.time() - start

    print(f"\nFirst call completed in {first_call_time:.2f}s")
    print(f"Result: {json.dumps(result, indent=2)}")

    print("\nStarting second call (warm container)...")
    start = time.time()
    result = service.health_check.remote()
    second_call_time = time.time() - start

    print(f"\nSecond call completed in {second_call_time:.2f}s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cold start:     {first_call_time:.2f}s")
    print(f"Warm container: {second_call_time:.2f}s")
    print(f"Speedup:        {first_call_time / second_call_time:.1f}x")
    print(f"Model load:     {result.get('load_time_seconds', 'N/A')}s")
