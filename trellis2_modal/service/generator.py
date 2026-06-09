"""
TRELLIS.2 Generator class for Modal deployment.

Handles model loading, image-to-3D generation, video rendering, and GLB extraction.
Imports are deferred because trellis2.* requires GPU and PYTHONPATH setup.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from PIL import Image

from .config import MODEL_NAME

# nvdiffrast has a maximum face count for rasterization
NVDIFFRAST_MAX_FACES = 16_777_216  # 2^24


class PipelineProtocol(Protocol):
    """Protocol defining the pipeline interface for type checking and mocking."""

    models: dict[str, Any]

    def to(self, device: Any) -> None: ...

    def cuda(self) -> None: ...

    def preprocess_image(self, image: "Image") -> "Image": ...

    def run(
        self,
        image: "Image",
        seed: int,
        sparse_structure_sampler_params: dict[str, Any],
        shape_slat_sampler_params: dict[str, Any],
        tex_slat_sampler_params: dict[str, Any],
        pipeline_type: str,
        max_num_tokens: int,
        **kwargs: Any,
    ) -> list[Any]: ...


def _default_pipeline_factory(model_name: str) -> PipelineProtocol:
    """Default factory that loads the real TRELLIS.2 pipeline."""
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    return Trellis2ImageTo3DPipeline.from_pretrained(model_name)


class TRELLIS2Generator:
    """GPU-accelerated TRELLIS.2 generator for Modal deployment."""

    def __init__(
        self,
        pipeline_factory: Callable[[str], PipelineProtocol] | None = None,
    ) -> None:
        """Initialize generator state. Model loaded via load_model()."""
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self.pipeline: PipelineProtocol | None = None
        self.envmap: Any = None
        self.load_time: float = 0.0

    @property
    def is_ready(self) -> bool:
        """Check if generator is fully ready for inference."""
        return self.pipeline is not None and self.envmap is not None

    def load_model(self) -> None:
        """Load model and HDRI. Called with @modal.enter()."""
        import cv2
        import torch
        from trellis2.renderers import EnvMap

        from .config import TRELLIS2_PATH

        start = time.perf_counter()

        # Load pipeline from HuggingFace
        try:
            self.pipeline = self._pipeline_factory(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TRELLIS.2 model '{MODEL_NAME}'. "
                f"Check HuggingFace cache and network connectivity."
            ) from e

        # Move pipeline to GPU
        self.pipeline.cuda()

        # Load HDRI for PBR rendering
        hdri_path = f"{TRELLIS2_PATH}/assets/hdri/forest.exr"
        hdri = cv2.imread(hdri_path, cv2.IMREAD_UNCHANGED)
        if hdri is None:
            raise RuntimeError(f"Failed to load HDRI from {hdri_path}")
        hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB)
        self.envmap = EnvMap(torch.tensor(hdri, dtype=torch.float32, device="cuda"))

        self.load_time = time.perf_counter() - start

    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory after operations."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # Running in test environment without torch

    def generate(
        self,
        image: "Image",
        seed: int = 42,
        pipeline_type: str = "1024_cascade",
        ss_params: dict[str, Any] | None = None,
        shape_params: dict[str, Any] | None = None,
        tex_params: dict[str, Any] | None = None,
        max_num_tokens: int = 49152,
    ) -> dict[str, Any]:
        """
        Generate MeshWithVoxel from image.

        Args:
            image: PIL Image (will be preprocessed)
            seed: Random seed for reproducibility
            pipeline_type: One of "512", "1024", "1024_cascade", "1536_cascade"
            ss_params: Sparse structure sampler params
            shape_params: Shape SLAT sampler params
            tex_params: Texture SLAT sampler params
            max_num_tokens: Maximum tokens for cascade (controls resolution cap)

        Returns:
            Packed state dict ready for serialization
        """
        from .state import pack_state

        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Ensure load_model() was called.")

        # Preprocess and generate
        processed = self.pipeline.preprocess_image(image)
        meshes = self.pipeline.run(
            processed,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=ss_params or {},
            shape_slat_sampler_params=shape_params or {},
            tex_slat_sampler_params=tex_params or {},
            max_num_tokens=max_num_tokens,
        )

        mesh = meshes[0]

        # Simplify to nvdiffrast limit
        mesh.simplify(NVDIFFRAST_MAX_FACES)

        state = pack_state(mesh)
        self._cleanup_gpu_memory()
        return state

    def render_preview_video(
        self,
        state: dict[str, Any],
        num_frames: int = 120,
        fps: int = 15,
    ) -> bytes:
        """
        Render PBR preview video from packed state.

        Returns MP4 video as bytes.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Generator not ready. Ensure load_model() was called successfully."
            )

        import imageio
        from trellis2.utils import render_utils

        from .state import unpack_state

        mesh = unpack_state(state)

        # Render with environment lighting
        raw_result = render_utils.render_video(
            mesh,
            resolution=512,
            num_frames=num_frames,
            envmap=self.envmap,
        )

        # Create PBR visualization frames
        frames = render_utils.make_pbr_vis_frames(raw_result, resolution=512)

        # Encode to MP4
        video_bytes = imageio.mimwrite(
            "<bytes>",
            frames,
            format="mp4",
            fps=fps,
        )

        self._cleanup_gpu_memory()
        return video_bytes

    def extract_glb(
        self,
        state: dict[str, Any],
        decimation_target: int = 1000000,
        texture_size: int = 4096,
        remesh: bool = True,
        remesh_band: float = 1.0,
        remesh_project: float = 0.0,
    ) -> bytes:
        """
        Extract GLB mesh from packed state.

        Args:
            state: Packed state from generate()
            decimation_target: Target vertex count (default 1M for quality)
            texture_size: Texture resolution (default 4096 for quality)
            remesh: Whether to remesh for cleaner topology
            remesh_band: Remesh band size
            remesh_project: Projection factor for remesh

        Returns:
            GLB file as bytes
        """
        if not self.is_ready:
            raise RuntimeError(
                "Generator not ready. Ensure load_model() was called successfully."
            )

        import io

        import torch
        from o_voxel.postprocess import to_glb

        from .state import unpack_state

        mesh = unpack_state(state)

        # Extract GLB using o_voxel
        glb_mesh = to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            aabb=torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device="cuda"),
            voxel_size=mesh.voxel_size,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=remesh_band,
            remesh_project=remesh_project,
            verbose=False,
        )

        # Export to bytes
        buffer = io.BytesIO()
        glb_mesh.export(buffer, file_type="glb")
        buffer.seek(0)
        glb_bytes = buffer.read()

        self._cleanup_gpu_memory()
        return glb_bytes
