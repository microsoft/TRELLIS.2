# TRELLIS.2 Modal Integration Plan

This document outlines the implementation plan for adding Modal serverless deployment to TRELLIS.2, following the patterns established in TRELLIS v1.

## Summary of Decisions

| Decision            | Choice                          | Rationale                                      |
|---------------------|---------------------------------|------------------------------------------------|
| State Serialization | Store Final MeshWithVoxel       | Simpler extraction, matches v1 pattern         |
| Attention Backend   | flash_attn                      | Best A100 performance, v2 default              |
| Preview Rendering   | Single PBR video + bundled HDRI | Quality/simplicity balance                     |
| CUDA Version        | 12.4 + PyTorch 2.6.0            | Matches upstream requirements                  |
| API Parameters      | Tiered (basic + advanced)       | Simple for basic use, flexible for power users |
| GLB Extraction      | Expose all parameters           | Maximum flexibility                            |
| GPU                 | A100-80GB                       | Sufficient for all pipeline types              |

## Architecture Overview

```
+-------------------------------------------------------------+
|  LOCAL GRADIO CLIENT                                        |
|  - Image upload, parameter controls, result display         |
|  - Stores LZ4-compressed MeshWithVoxel state                |
|  - Communicates via HTTPS                                   |
+-----------------------------+-------------------------------+
                              |
                              | POST /generate, POST /extract_glb
                              | Headers: X-API-Key: sk_xxx
                              v
+-------------------------------------------------------------+
|  MODAL WEB ENDPOINTS (A100-80GB)                            |
|                                                             |
|  Authentication Layer (API Key validation from Volume)      |
|                           |                                 |
|                           v                                 |
|  +-------------------------------------------------------+  |
|  |  TRELLIS2Service Class                                |  |
|  |  gpu="A100-80GB"                                      |  |
|  |  enable_memory_snapshot=True                          |  |
|  |                                                       |  |
|  |  @enter(snap=True): Load pipeline to CPU              |  |
|  |  @enter(snap=False): Move pipeline to GPU             |  |
|  |  generate(): Image -> MeshWithVoxel state + video     |  |
|  |  extract_glb(): State -> GLB mesh                     |  |
|  +-------------------------------------------------------+  |
|                                                             |
|  Volumes:                                                   |
|  - /cache/huggingface: Model weights cache                  |
|  - /data/keys.json: API key storage                         |
+-------------------------------------------------------------+
```

## Implementation Phases

### Phase 1: Modal Image Definition (image.py)

**Goal**: Build a Modal container image with all CUDA extensions pre-compiled.

**Test First (TDD)**:
```python
# tests/test_image_build.py
def test_image_has_required_cuda_extensions():
    """Verify all CUDA extensions are importable."""
    # This test runs inside the Modal container

def test_pytorch_cuda_available():
    """Verify PyTorch can access CUDA."""

def test_flash_attn_available():
    """Verify flash_attn is installed and working."""

def test_trellis2_pipeline_importable():
    """Verify trellis2 package is on PYTHONPATH."""
```

**Implementation**:

1. Base image: `nvidia/cuda:12.4.0-devel-ubuntu22.04` + Python 3.10
2. System packages: git, ninja-build, cmake, clang, OpenGL libs
3. Python packages (no GPU needed):
   - pillow, imageio, imageio-ffmpeg, tqdm, easydict
   - opencv-python-headless, trimesh, transformers, huggingface-hub
   - lz4, fastapi, safetensors, kornia, timm
4. GPU build block (requires `gpu="T4"` for compilation):
   - PyTorch 2.6.0 + CUDA 12.4
   - flash-attn 2.7.3
   - nvdiffrast (v0.4.0 tag)
   - nvdiffrec (renderutils branch from JeffreyXiang fork)
   - CuMesh (from JeffreyXiang/CuMesh)
   - FlexGEMM (from JeffreyXiang/FlexGEMM)
   - o-voxel (bundled in repo, copy and build)
5. Clone TRELLIS.2 repo to /opt/TRELLIS.2
6. Pre-download models:
   - DINOv2 (via torch.hub)
   - BiRefNet for background removal
7. Bundle HDRI file (forest.exr) for rendering
8. Environment variables:
   ```
   ATTN_BACKEND=flash_attn
   PYTHONPATH=/opt/TRELLIS.2
   HF_HOME=/cache/huggingface
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   OPENCV_IO_ENABLE_OPENEXR=1
   ```

**Verification**: `modal run trellis2_modal/service/image.py::verify_image`

---

### Phase 2: Generator Wrapper (generator.py)

**Goal**: Wrap Trellis2ImageTo3DPipeline with Modal's two-phase loading pattern.

**Test First (TDD)**:
```python
# tests/test_generator.py
def test_generator_load_model_cpu():
    """Test CPU loading with mock pipeline factory."""

def test_generator_move_model_gpu():
    """Test GPU transfer after CPU load."""

def test_generator_generate_returns_mesh_with_voxel():
    """Test generation returns correct state structure."""

def test_generator_all_pipeline_types():
    """Test 512, 1024, 1024_cascade, 1536_cascade."""

def test_generator_render_preview_video():
    """Test PBR video rendering with HDRI."""
```

**Implementation**:

```python
class TRELLIS2Generator:
    def __init__(self, pipeline_factory=None):
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self.pipeline = None
        self.envmap = None

    def load_model_cpu(self):
        """@modal.enter(snap=True) - Load to CPU for snapshot."""
        self.pipeline = self._pipeline_factory(MODEL_NAME)
        # Don't load envmap here - needs GPU

    def move_model_gpu(self):
        """@modal.enter(snap=False) - Move to GPU after restore."""
        self.pipeline.cuda()
        self._load_envmap()

    def _load_envmap(self):
        """Load bundled HDRI for rendering."""
        import cv2
        from trellis2.renderers import EnvMap
        hdri = cv2.imread('/opt/TRELLIS.2/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED)
        hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB)
        self.envmap = EnvMap(torch.tensor(hdri, dtype=torch.float32, device='cuda'))

    def generate(self, image, seed, pipeline_type,
                 ss_params=None, shape_params=None, tex_params=None):
        """Generate MeshWithVoxel from image."""
        mesh = self.pipeline.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=ss_params or {},
            shape_slat_sampler_params=shape_params or {},
            tex_slat_sampler_params=tex_params or {},
        )[0]
        mesh.simplify(16777216)  # nvdiffrast limit
        return mesh

    def render_preview_video(self, mesh, num_frames=120, fps=15):
        """Render PBR video with environment lighting."""
        from trellis2.utils import render_utils
        frames = render_utils.render_video(mesh, envmap=self.envmap,
                                           num_frames=num_frames)
        # Encode to MP4 bytes
        ...
```

---

### Phase 3: State Serialization (state.py)

**Goal**: Pack/unpack MeshWithVoxel for client-side storage.

**Test First (TDD)**:
```python
# tests/test_state.py
def test_pack_state_contains_all_fields():
    """Verify packed state has vertices, faces, attrs, coords, etc."""

def test_unpack_state_reconstructs_mesh():
    """Verify unpacked mesh matches original."""

def test_state_roundtrip_preserves_data():
    """Pack -> unpack -> pack produces identical state."""

def test_state_is_json_serializable_after_numpy():
    """Verify numpy arrays are used (not tensors)."""
```

**Implementation**:

```python
def pack_state(mesh: MeshWithVoxel) -> dict:
    """Pack MeshWithVoxel into serializable dict."""
    return {
        "vertices": mesh.vertices.cpu().numpy(),
        "faces": mesh.faces.cpu().numpy(),
        "attrs": mesh.attrs.cpu().numpy(),
        "coords": mesh.coords.cpu().numpy(),
        "voxel_size": float(mesh.voxel_size),
        "voxel_shape": list(mesh.voxel_shape),
        "origin": mesh.origin.cpu().tolist(),
        "layout": {k: [v.start, v.stop] for k, v in mesh.layout.items()},
    }

def unpack_state(state: dict) -> MeshWithVoxel:
    """Reconstruct MeshWithVoxel from packed state."""
    layout = {k: slice(v[0], v[1]) for k, v in state["layout"].items()}
    return MeshWithVoxel(
        vertices=torch.tensor(state["vertices"], device="cuda"),
        faces=torch.tensor(state["faces"], device="cuda"),
        origin=state["origin"],
        voxel_size=state["voxel_size"],
        coords=torch.tensor(state["coords"], device="cuda"),
        attrs=torch.tensor(state["attrs"], device="cuda"),
        voxel_shape=torch.Size(state["voxel_shape"]),
        layout=layout,
    )
```

---

### Phase 4: Service Endpoints (service.py)

**Goal**: Modal web endpoints for generate and extract_glb.

**Test First (TDD)**:
```python
# tests/test_service.py
def test_generate_endpoint_validates_api_key():
    """401 on missing/invalid key."""

def test_generate_endpoint_validates_image():
    """400 on invalid image payload."""

def test_generate_endpoint_returns_state_and_video():
    """Successful generation returns expected structure."""

def test_extract_glb_endpoint_returns_glb():
    """GLB extraction returns valid GLB bytes."""

def test_health_endpoint_no_auth_required():
    """Health check works without API key."""
```

**Implementation**:

```python
@app.cls(
    image=trellis2_image,
    gpu="A100-80GB",
    volumes={
        HF_CACHE_PATH: hf_cache_volume,
        API_KEYS_PATH: api_keys_volume,
    },
    enable_memory_snapshot=True,
    container_idle_timeout=300,
)
class TRELLIS2Service:
    @modal.enter(snap=True)
    def load_model_cpu(self):
        self.generator = TRELLIS2Generator()
        self.generator.load_model_cpu()

    @modal.enter(snap=False)
    def load_model_gpu(self):
        self.generator.move_model_gpu()

    @modal.web_endpoint(method="POST")
    def generate(self, request: Request) -> dict:
        """Generate 3D from image."""
        # Auth check
        # Validate request
        # Generate mesh
        # Render video
        # Pack state
        # Return base64 state + video

    @modal.web_endpoint(method="POST")
    def extract_glb(self, request: Request) -> dict:
        """Extract GLB from state."""
        # Auth check
        # Unpack state
        # Call o_voxel.postprocess.to_glb()
        # Return base64 GLB

    @modal.web_endpoint(method="GET")
    def health(self) -> dict:
        return {"status": "ok", "service": "trellis2-api"}
```

**API Request/Response**:

```
POST /generate
Request:
{
  "image": "<base64-png>",
  "seed": 42,
  "pipeline_type": "1024_cascade",  // 512, 1024, 1024_cascade, 1536_cascade

  // Optional advanced params
  "ss_sampling_steps": 12,
  "ss_guidance_strength": 7.5,
  "ss_guidance_rescale": 0.7,
  "ss_rescale_t": 5.0,
  "shape_slat_sampling_steps": 12,
  "shape_slat_guidance_strength": 7.5,
  "shape_slat_guidance_rescale": 0.5,
  "shape_slat_rescale_t": 3.0,
  "tex_slat_sampling_steps": 12,
  "tex_slat_guidance_strength": 1.0,
  "tex_slat_guidance_rescale": 0.0,
  "tex_slat_rescale_t": 3.0
}

Response:
{
  "state": "<base64-lz4-compressed>",
  "video": "<base64-mp4>"
}
```

```
POST /extract_glb
Request:
{
  "state": "<base64-lz4-compressed>",
  "decimation_target": 500000,
  "texture_size": 2048,
  "remesh": true,
  "remesh_band": 1.0,
  "remesh_project": 0.0
}

Response:
{
  "glb": "<base64-glb>"
}
```

---

### Phase 5: Client (api.py, app.py)

**Goal**: Gradio client that talks to Modal service.

**Test First (TDD)**:
```python
# tests/test_api_client.py
def test_client_generate_sends_correct_request():
    """Verify request format matches API spec."""

def test_client_handles_cold_start_timeout():
    """Retry logic for slow cold starts."""

def test_client_decompresses_state():
    """LZ4 decompression works correctly."""
```

**Implementation**:

Adapt v1's client with updated:
- Parameter names for TRELLIS.2
- Pipeline type selector (512/1024/1536)
- Advanced parameter accordions for each stage
- Updated preview display for PBR video

---

### Phase 6: Auth & Compression (reuse from v1)

**Goal**: Copy and adapt auth.py and compression.py from v1.

These modules are largely unchanged:
- `auth.py`: API key validation, rate limiting, quota management
- `compression.py`: LZ4 compression for state transfer

Minor updates:
- Update app name references
- Update volume names

---

### Phase 7: Documentation & CI

**Goal**: Docs and GitHub Actions for testing.

Files to create:
- `trellis2_modal/docs/MODAL_INTEGRATION.md`
- `trellis2_modal/docs/OPERATIONS_RUNBOOK.md`
- `.github/workflows/modal-ci.yml`

---

## File Structure

```
TRELLIS.2/
├── trellis2/                          # Existing - unchanged
├── o-voxel/                           # Existing - unchanged
├── trellis2_modal/                    # NEW
│   ├── __init__.py
│   ├── service/
│   │   ├── __init__.py
│   │   ├── config.py                  # GPU type, model name, defaults
│   │   ├── image.py                   # Modal image definition
│   │   ├── generator.py               # TRELLIS2Generator class
│   │   ├── service.py                 # Modal endpoints
│   │   ├── state.py                   # Pack/unpack MeshWithVoxel
│   │   ├── auth.py                    # API key management (from v1)
│   │   └── streaming.py               # SSE utilities (from v1)
│   ├── client/
│   │   ├── __init__.py
│   │   ├── app.py                     # Gradio UI
│   │   ├── api.py                     # HTTP client
│   │   ├── compression.py             # LZ4 (from v1)
│   │   └── requirements.txt
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                # Mock fixtures
│   │   ├── test_generator.py
│   │   ├── test_state.py
│   │   ├── test_api_client.py
│   │   ├── test_auth.py
│   │   ├── test_compression.py
│   │   ├── test_service.py
│   │   └── test_integration.py
│   └── docs/
│       ├── MODAL_INTEGRATION.md
│       └── OPERATIONS_RUNBOOK.md
├── .github/
│   └── workflows/
│       └── modal-ci.yml
└── PLAN.md                            # This file
```

---

## Dependencies Comparison

### TRELLIS v1 (CUDA 11.8)
```
torch==2.4.0+cu118
xformers==0.0.27.post2
spconv-cu118
nvdiffrast (pinned commit)
diffoctreerast (pinned commit)
diff-gaussian-rasterization (mip-splatting)
```

### TRELLIS.2 (CUDA 12.4)
```
torch==2.6.0+cu124
flash-attn==2.7.3
nvdiffrast==0.4.0
nvdiffrec (renderutils branch)
cumesh (JeffreyXiang/CuMesh)
flex_gemm (JeffreyXiang/FlexGEMM)
o_voxel (bundled)
```

---

## Estimated Effort

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Modal Image | 4-6 hours (CUDA builds are finicky) |
| 2 | Generator | 2-3 hours |
| 3 | State Serialization | 1-2 hours |
| 4 | Service Endpoints | 3-4 hours |
| 5 | Client | 2-3 hours |
| 6 | Auth & Compression | 1 hour (mostly copy) |
| 7 | Docs & CI | 2 hours |

**Total**: ~16-21 hours (2-3 days)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CUDA extension build failures | Pin exact commits, test builds early |
| Memory issues on A100-80GB | low_vram mode is built-in to TRELLIS.2 |
| Cold start too slow | GPU memory snapshots (proven in v1) |
| HDRI file too large for image | Compress or use smaller resolution |
| State too large after compression | Already using LZ4, can tune if needed |

---

## Verification Checklist

After each phase, verify:

- [ ] All tests pass locally (mocked, no GPU)
- [ ] `modal run` works for smoke tests
- [ ] `modal deploy` succeeds
- [ ] Health endpoint responds
- [ ] Generate endpoint produces valid output
- [ ] Extract GLB produces downloadable file
- [ ] Cold start time acceptable (<30s with snapshot)
- [ ] Warm request latency acceptable (<60s for 1024)

---

## Next Steps

1. Review and approve this plan
2. Start Phase 1: Modal Image Definition
3. Test image builds before proceeding
4. Continue through phases sequentially
