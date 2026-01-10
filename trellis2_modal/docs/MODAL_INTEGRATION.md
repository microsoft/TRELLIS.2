# TRELLIS.2 Modal Integration

This guide explains how to deploy and use TRELLIS.2 on Modal's serverless GPU infrastructure.

## Prerequisites

- Python 3.11+
- [Modal account](https://modal.com/) with GPU quota (A100-80GB)
- Modal CLI installed and authenticated
- [HuggingFace account](https://huggingface.co/) with access to gated models

```bash
pip install -r trellis2_modal/requirements-deploy.txt
modal token new
```

### HuggingFace Setup

TRELLIS.2 uses gated models that require HuggingFace authentication:

1. **Accept model licenses** on HuggingFace (click "Agree and access repository"):
   - [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) - Main TRELLIS.2 model
   - [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) - DINOv3 vision encoder (used for image conditioning)
   - [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) - Background removal model (CC BY-NC 4.0, **non-commercial use only**)

   **Important**: All three models must be accepted. The DINOv3 and RMBG-2.0 models are
   loaded automatically by the TRELLIS.2 pipeline.

2. **Create a HuggingFace token** at https://huggingface.co/settings/tokens
   - Use "Read" access (write not needed)

3. **Verify access** (should return model info, not 403):
   ```bash
   # Check all three gated models
   curl -H "Authorization: Bearer hf_your_token" \
     https://huggingface.co/api/models/microsoft/TRELLIS.2-4B
   curl -H "Authorization: Bearer hf_your_token" \
     https://huggingface.co/api/models/facebook/dinov3-vitl16-pretrain-lvd1689m
   curl -H "Authorization: Bearer hf_your_token" \
     https://huggingface.co/api/models/briaai/RMBG-2.0
   ```

4. **Create Modal secret**:
   ```bash
   modal secret create huggingface HF_TOKEN=hf_your_token_here
   ```

## Project Structure

```
trellis2_modal/
├── requirements-deploy.txt  # Dependencies for modal deploy
├── service/           # Modal service (runs on cloud)
│   ├── config.py      # Configuration constants
│   ├── image.py       # Container image definition
│   ├── generator.py   # TRELLIS2Generator wrapper
│   ├── service.py     # Web endpoints
│   ├── state.py       # MeshWithVoxel serialization
│   └── auth.py        # API key authentication
├── client/            # Local client (runs on your machine)
│   ├── api.py         # HTTP client
│   ├── app.py         # Gradio UI
│   ├── compression.py # State compression
│   └── requirements.txt  # Client dependencies
├── tests/             # Unit tests
└── docs/              # This documentation
```

## Deployment

### 1. Verify the Image Builds

First, verify all CUDA extensions compile correctly:

```bash
modal run trellis2_modal/service/image.py
```

This runs a verification that checks:
- CUDA availability and version
- flash_attn, nvdiffrast, cumesh, flex_gemm extensions
- TRELLIS.2 pipeline imports
- HDRI files exist

Expected output:
```
✓ Image verification PASSED
```

### 2. Deploy the Service

Deploy to Modal:

```bash
modal deploy -m trellis2_modal.service.service
```

Note the endpoint URLs printed:
```
├── Created web endpoint for TRELLIS2Service.health => https://your-app--health.modal.run
├── Created web endpoint for TRELLIS2Service.generate => https://your-app--generate.modal.run
└── Created web endpoint for TRELLIS2Service.extract_glb => https://your-app--extract-glb.modal.run
```

### 3. Create Proxy Auth Tokens

Authentication uses Modal Proxy Auth Tokens. Create tokens in the Modal dashboard:

1. Go to https://modal.com/settings/proxy-auth-tokens
2. Click "New Token"
3. Copy both the **Token ID** (Modal-Key) and **Token Secret** (Modal-Secret)
   - ⚠️ The secret is only shown once - save it immediately!

Store credentials locally (choose one method):

**Option A: Environment variables**
```bash
export TRELLIS2_MODAL_KEY="wk-xxxxx"
export TRELLIS2_MODAL_SECRET="ws-xxxxx"
```

**Option B: Secrets file** (`~/.trellis2_modal_secrets.json`)
```json
{
    "modal_key": "wk-xxxxx",
    "modal_secret": "ws-xxxxx"
}
```

The file is in your home directory and is gitignored.

### 4. Test the Deployment

Test the health endpoint (no auth required):

```bash
curl https://your-app--health.modal.run
# {"status": "ok", "service": "trellis2-api"}
```

Test generation (auth required):

```bash
curl -X POST https://your-app--generate.modal.run \
  -H "Modal-Key: wk-xxxxx" \
  -H "Modal-Secret: ws-xxxxx" \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>", "seed": 42}'
```

## Running the Client

### Environment Setup

The client loads credentials from (in priority order):
1. Environment variables: `TRELLIS2_MODAL_KEY` and `TRELLIS2_MODAL_SECRET`
2. Secrets file: `~/.trellis2_modal_secrets.json`

```bash
export TRELLIS2_API_URL=https://your-app--generate.modal.run
export TRELLIS2_MODAL_KEY=wk-xxxxx
export TRELLIS2_MODAL_SECRET=ws-xxxxx
```

The client auto-detects Modal subdomain routing from the URL pattern and derives the `extract_glb` endpoint automatically.

### Launch Gradio UI

```bash
pip install -r trellis2_modal/client/requirements.txt
python -m trellis2_modal.client.app
```

Open http://localhost:7860 in your browser.

### Programmatic Usage

```python
from trellis2_modal.client import TRELLIS2APIClient

# Credentials loaded automatically from env vars or ~/.trellis2_modal_secrets.json
client = TRELLIS2APIClient(base_url="https://your-app--generate.modal.run")

# Or provide credentials explicitly:
# client = TRELLIS2APIClient(
#     base_url="https://your-app--generate.modal.run",
#     modal_key="wk-xxxxx",
#     modal_secret="ws-xxxxx",
# )

# Generate 3D from image (uses defaults)
result = client.generate(image_path="input.png")

# result contains:
# - state: compressed state for GLB extraction
# - video: base64-encoded MP4 preview

# Extract GLB with default settings (high quality)
client.extract_glb(
    state=result["state"],
    output_path="output.glb",
)

# Or for game engines with polygon limits:
client.extract_glb(
    state=result["state"],
    output_path="game_asset.glb",
    decimation_target=10000,   # ~10K triangles
    texture_size=1024,
)
```

## Complete API Reference

### `client.generate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | *required* | Path to input image (PNG, JPEG) |
| `seed` | int | 42 | Random seed for reproducibility |
| `pipeline_type` | str | "1024_cascade" | Resolution pipeline (see below) |
| `ss_sampling_steps` | int | 12 | Sparse structure denoising steps |
| `ss_guidance_strength` | float | 7.5 | Sparse structure guidance |
| `shape_slat_sampling_steps` | int | 12 | Shape refinement denoising steps |
| `shape_slat_guidance_strength` | float | 7.5 | Shape refinement guidance |
| `tex_slat_sampling_steps` | int | 12 | Texture generation denoising steps |
| `tex_slat_guidance_strength` | float | 1.0 | Texture generation guidance |

#### Understanding the Three-Stage Pipeline

TRELLIS.2 generates 3D models in three stages:

1. **Sparse Structure (SS)**: Creates the initial coarse 3D voxel structure from the input image. This determines the overall shape and proportions.

2. **Shape SLAT (Structured Latent)**: Refines the geometry to the target resolution, adding fine geometric details like edges, corners, and surface features.

3. **Texture SLAT**: Generates PBR (Physically-Based Rendering) materials including base color, metallic, roughness, and opacity.

#### Input Image Guidelines

Quality of results depends heavily on input image preparation:

**Optimal inputs:**
- Clean, well-lit subject against neutral/solid background
- Object centered in frame with minimal perspective distortion
- Resolution at least 512×512 (1024×1024 recommended)
- Clear separation between subject and background
- Consistent lighting without harsh shadows or specular highlights

**Problematic inputs:**
- Cluttered backgrounds (confuses edge detection)
- Extreme perspective angles or fish-eye distortion
- Low resolution or heavily compressed images
- Transparent or highly reflective surfaces
- Multiple overlapping objects

#### Parameter Tuning Guide

**Sampling Steps** (`*_sampling_steps`):
- Controls quality vs speed tradeoff
- Default of 12 is well-balanced
- 8 steps: Faster but may have artifacts
- 16-20 steps: Higher quality, diminishing returns beyond
- Each additional step adds ~1-2 seconds

**Guidance Strength** (`*_guidance_strength`):
- Controls how closely output follows the input image
- **Shape stages (7.5 default)**: Higher values produce more faithful reconstructions
  - 5.0-7.5: Good balance of accuracy and natural appearance
  - 10.0-15.0: Very literal interpretation, may cause rigidity
  - Below 3.0: More creative but may drift from input
- **Texture stage (1.0 default)**: Lower values allow natural material variation
  - 1.0: Natural-looking materials
  - 3.0-5.0: More constrained to input colors
  - Higher: May produce flat, unnatural textures

#### Tuning by Object Type

| Object Type | SS Guidance | Shape Guidance | Notes |
|-------------|-------------|----------------|-------|
| **Hard-surface** (furniture, vehicles, architecture) | 8-9 | 7.5 | Stricter geometric adherence |
| **Organic** (characters, plants, fabric) | 7.5 | 7.5 | Default values work well |
| **Ambiguous shapes** | 5-7 | 5-7 | Lower guidance for coherence |

**Tip**: Use the `seed` parameter for reproducibility. When you find settings that work well for a particular object type, record the seed to generate consistent results across similar inputs.

### `client.extract_glb()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state` | str | *required* | Base64 state from `generate()` |
| `output_path` | str | *required* | Where to save the GLB file |
| `decimation_target` | int | 1,000,000 | Target triangle count |
| `texture_size` | int | 4096 | Texture resolution (512/1024/2048/4096) |
| `remesh` | bool | True | Clean up mesh topology |
| `remesh_band` | float | 1.0 | Remesh band size |
| `remesh_project` | float | 0.0 | Remesh projection factor |

### GLB Extraction Presets

| Use Case | decimation_target | texture_size | Approx Size |
|----------|-------------------|--------------|-------------|
| **Maximum Quality** | 1,000,000 | 4096 | ~30MB |
| **Web Viewers / Sketchfab** | 100,000 | 2048 | ~5MB |
| **Game Engines (Unity/Unreal/Godot)** | 10,000 | 1024 | ~1MB |
| **Mobile / AR** | 5,000 | 512 | ~300KB |
| **Low-poly / Stylized** | 2,000 | 512 | ~150KB |

**Platform-specific notes**:
- **Unity/Unreal**: GLB imports directly; may need material adjustment for opacity
- **Godot**: Native GLB support with PBR materials
- **Web (Three.js, Babylon.js)**: Use Web/Viewer preset for good balance
- **Some platforms require FBX**: Convert via Blender if GLB not supported

## Configuration

### GPU Type

Default: `A100-80GB` (configured in `config.py`)

The A100-80GB provides:
- Sufficient VRAM for all pipeline types
- Good balance of cost and performance

### Pipeline Types

| Type | Resolution | Time (A100) | Use Case |
|------|------------|-------------|----------|
| `512` | 512³ | ~10s | Fast preview |
| `1024` | 1024³ | ~25s | Good quality |
| `1024_cascade` | 512→1024 | ~30s | High quality (recommended) |
| `1536_cascade` | 512→1536 | ~90s | Maximum quality |

## Cold Starts

TRELLIS.2 has a cold start time of approximately **2-2.5 minutes** on A100-80GB:
- Container startup: ~20s
- Model loading: ~2 minutes (loading from HuggingFace cache + GPU initialization)

### Memory Snapshots: Not Recommended

Both CPU and GPU Memory Snapshots were tested and found **not effective** for TRELLIS.2:

| Configuration | Cold Start | Model Load | Improvement |
|---------------|------------|------------|-------------|
| No snapshots (baseline) | ~143s | ~124s | — |
| GPU Memory Snapshots | ~146s | ~131s | <5% |

**Root cause**: TRELLIS.2's dependencies (flex_gemm, Triton, flash_attn) require
re-initialization after snapshot restoration, negating the benefits. The model
weights may be restored, but GPU state for these libraries is not preserved.

### Recommended Cold Start Strategies

1. **Keep containers warm** (default, recommended):
   - `scaledown_window=300` keeps containers alive for 5 minutes after last request
   - Cost: ~$0.10/hour for idle A100-80GB container

2. **Periodic warm-up ping** (for consistent response times):
   - Ping the service every 4 minutes to prevent scale-down
   - Use a simple health check: `curl https://your-app--health.modal.run`

3. **min_containers=1** (for always-on availability):
   - Keeps one container always running
   - Cost: ~$2.40/hour (continuous A100-80GB)

## Troubleshooting

See [OPERATIONS_RUNBOOK.md](./OPERATIONS_RUNBOOK.md) for common issues and solutions.

## Cost Estimation

| Operation | GPU Time | Approx Cost |
|-----------|----------|-------------|
| Cold start | 2-3 min | ~$0.10-0.15 |
| Generate (1024_cascade) | 30s | ~$0.02 |
| Extract GLB | 30-60s | ~$0.02-0.04 |

Tips to reduce costs:
- Use `container_idle_timeout=300` to keep containers warm between requests
- Use lower resolution pipelines for previews
- Use smaller GLB presets when file size matters
