# TRELLIS.2 on Modal

Deploy TRELLIS.2 to Modal's serverless GPU infrastructure. Generate 3D assets from images without needing a local GPU.

**Why use this?**
- No local GPU required - runs on cloud A100s
- Pay only for what you use (~$0.05 per generation)
- Access via web UI or API from any device

## What You Get

TRELLIS.2 generates 3D meshes with full PBR (Physically-Based Rendering) materials:
- **Base Color** - diffuse texture
- **Metallic** - metal vs non-metal surfaces
- **Roughness** - shiny vs matte appearance
- **Opacity** - transparency and translucency support

The model uses DINOv2 vision features to infer geometry even for occluded areas not visible in the input image. It handles complex topology including open surfaces (clothing, leaves), non-manifold geometry, and internal structures.

## Prerequisites

Before starting, you need accounts and tokens from three services:

| Service | What you need | Where to get it |
|---------|--------------|-----------------|
| **Modal** | Account + A100 GPU quota | [modal.com](https://modal.com/) |
| **HuggingFace** | Account + accept 3 model licenses | See below |
| **Modal Proxy Auth** | Token ID + Secret | Modal dashboard |

### HuggingFace Model Access

TRELLIS.2 requires three gated models. Click each link and accept the license:

1. [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) - Main model
2. [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) - Vision encoder
3. [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) - Background removal *(non-commercial)*

Then create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read access).

## Quick Start

### 1. Install Modal CLI

```bash
pip install modal
modal token new
```

### 2. Add HuggingFace Token to Modal

```bash
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

### 3. Deploy the Service

```bash
cd TRELLIS.2
pip install -r trellis2_modal/requirements-deploy.txt
modal deploy -m trellis2_modal.service.service
```

Note the endpoint URLs printed (e.g., `https://yourname--trellis2-3d-...generate.modal.run`).

### 4. Create API Credentials

1. Go to [modal.com/settings/proxy-auth-tokens](https://modal.com/settings/proxy-auth-tokens)
2. Click "New Token"
3. Save both the Token ID and Secret (secret shown only once!)

Store them locally:
```bash
cat > ~/.trellis2_modal_secrets.json << 'EOF'
{
    "modal_key": "wk-xxxxx",
    "modal_secret": "ws-xxxxx"
}
EOF
```

### 5. Run the Web UI

```bash
pip install -r trellis2_modal/client/requirements.txt
export TRELLIS2_API_URL=https://yourname--trellis2-3d-trellis2service-generate.modal.run
python -m trellis2_modal.client.app
```

Open http://localhost:7860, upload an image, and click Generate.

> **First request takes 2-3 minutes** (cold start). Subsequent requests take 10-90 seconds depending on resolution.

## Input Image Guidelines

Quality of results depends heavily on input image preparation:

**Optimal inputs:**
- Clean, well-lit subject against neutral/solid background
- Object centered in frame with minimal perspective distortion
- Resolution at least 512×512 (1024×1024 recommended)
- Clear separation between subject and background
- Consistent lighting without harsh shadows

**Problematic inputs:**
- Cluttered backgrounds (confuses edge detection)
- Extreme perspective angles or fish-eye distortion
- Low resolution or heavily compressed images
- Transparent or highly reflective surfaces
- Multiple overlapping objects

For product photography, a simple white or gray background significantly improves results.

## API Usage

### Basic Example

```python
from trellis2_modal.client import TRELLIS2APIClient

client = TRELLIS2APIClient(
    base_url="https://yourname--trellis2-3d-trellis2service-generate.modal.run"
)

# Generate 3D from image
result = client.generate(image_path="input.png", pipeline_type="1024_cascade")

# Extract GLB mesh (high quality)
client.extract_glb(state=result["state"], output_path="output.glb")
```

Credentials are loaded automatically from `~/.trellis2_modal_secrets.json` or environment variables.

### Export for Different Platforms

Adjust `decimation_target` and `texture_size` based on your target platform:

```python
# High-quality for rendering, web viewers, Sketchfab
client.extract_glb(
    state=result["state"],
    output_path="high_quality.glb",
    decimation_target=100000,
    texture_size=2048,
)

# Game engines (Unity, Unreal, Godot)
client.extract_glb(
    state=result["state"],
    output_path="game_asset.glb",
    decimation_target=10000,
    texture_size=1024,
)

# Mobile games, AR apps
client.extract_glb(
    state=result["state"],
    output_path="mobile_asset.glb",
    decimation_target=5000,
    texture_size=512,
)
```

## Generation Parameters

### Pipeline Types

Controls the output resolution. Higher resolution = more detail but slower.

| Pipeline | Resolution | Time | Use Case |
|----------|------------|------|----------|
| `512` | 512³ voxels | ~10s | Quick preview, iteration |
| `1024_cascade` | 1024³ voxels | ~30s | **Recommended** for most uses |
| `1536_cascade` | 1536³ voxels | ~90s | Maximum quality, hero assets |

### Quality Tuning

TRELLIS.2 generates 3D in three stages, each with tunable parameters:

| Stage | What it does | Parameters |
|-------|--------------|------------|
| **Sparse Structure** | Creates initial 3D shape from image | `ss_sampling_steps`, `ss_guidance_strength` |
| **Shape Refinement** | Adds geometric detail | `shape_slat_sampling_steps`, `shape_slat_guidance_strength` |
| **Texture Generation** | Applies PBR materials | `tex_slat_sampling_steps`, `tex_slat_guidance_strength` |

**Parameter meanings:**

- **`*_sampling_steps`** (default: 12): Number of denoising iterations. Higher = better quality but slower. Range: 8-20 is reasonable; beyond 20 has diminishing returns. Each additional step adds ~1-2 seconds.

- **`*_guidance_strength`**: How closely to follow the input image.
  - Shape stages default to **7.5** — produces faithful reconstructions
  - Texture stage defaults to **1.0** — allows natural material variation
  - Higher values (10-15) = more literal interpretation, risk of artifacts
  - Lower values (1-3) = more creative freedom, may drift from input

### Tuning by Object Type

| Object Type | SS Guidance | Shape Guidance | Notes |
|-------------|-------------|----------------|-------|
| **Hard-surface** (furniture, vehicles, architecture) | 8-9 | 7.5 | Stricter geometric adherence |
| **Organic** (characters, plants, fabric) | 7.5 | 7.5 | Default values work well |
| **Ambiguous shapes** | 5-7 | 5-7 | Lower guidance for coherence |

**Example with custom parameters:**

```python
result = client.generate(
    image_path="input.png",
    pipeline_type="1024_cascade",
    seed=42,  # For reproducibility
    # Increase steps for higher quality (slower)
    ss_sampling_steps=16,
    shape_slat_sampling_steps=16,
    tex_slat_sampling_steps=16,
    # Adjust guidance based on object type
    ss_guidance_strength=8.0,       # Higher for hard-surface
    shape_slat_guidance_strength=7.5,
    tex_slat_guidance_strength=1.0,
)
```

**Tip**: Use the `seed` parameter for reproducibility. When you find settings that work well, record the seed to generate consistent results.

## GLB Extraction Options

After generation, extract a mesh with these options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decimation_target` | 1,000,000 | Target triangle count. Lower = smaller file, less detail. |
| `texture_size` | 4096 | Texture resolution: 512, 1024, 2048, or 4096 |
| `remesh` | True | Clean up mesh topology (recommended) |

### Recommended Presets by Use Case

| Use Case | decimation_target | texture_size | File Size |
|----------|-------------------|--------------|-----------|
| **Maximum Quality** | 1,000,000 | 4096 | ~30MB |
| **Web Viewers / Sketchfab** | 100,000 | 2048 | ~5MB |
| **Game Engines** | 10,000 | 1024 | ~1MB |
| **Mobile / AR** | 5,000 | 512 | ~300KB |
| **Low-poly Stylized** | 2,000 | 512 | ~150KB |

**Note**: Aggressive decimation may collapse fine details like fingers, facial features, or thin structural elements. Test with your specific asset types.

## Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Missing geometry / holes | Poor background separation in input | Use cleaner background, better lighting |
| Distorted mesh | Perspective distortion in input | Re-photograph with less distortion |
| Low-quality textures | Low input resolution or small texture_size | Use higher resolution input (1024×1024+), increase texture_size |
| 403 from HuggingFace | Model licenses not accepted | Accept all three model licenses (links above) |
| Cold start every time | Container scaling to zero | Ping health endpoint every 4 minutes to keep warm |
| CUDA OOM | Insufficient GPU memory | Use lower resolution pipeline or reduce texture_size |

## Known Limitations

- **Small holes in geometry**: Generated meshes may occasionally have minor topological artifacts. For watertight meshes (e.g., 3D printing), post-processing in Blender may be needed.
- **Style variation**: This is a base model without aesthetic fine-tuning. Results reflect training data distribution.
- **Opacity in GLB**: Transparency is preserved in texture alpha channel but may need manual material setup in some applications.

## Cost

| What | GPU Time | Cost |
|------|----------|------|
| Cold start | 2-3 min | ~$0.10 |
| Generation (1024) | 30s | ~$0.02 |
| GLB extraction | 30-60s | ~$0.03 |

Containers stay warm for 5 minutes between requests (no cold start cost for sequential generations).

## Full Documentation

See [docs/MODAL_INTEGRATION.md](docs/MODAL_INTEGRATION.md) for:
- Complete API reference
- Operations runbook
- Cold start optimization strategies
