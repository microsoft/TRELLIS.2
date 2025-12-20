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

Deploy to Modal with memory snapshots enabled:

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
client = TRELLIS2APIClient(base_url="https://your-app.modal.run")

# Or provide credentials explicitly:
# client = TRELLIS2APIClient(
#     base_url="https://your-app.modal.run",
#     modal_key="wk-xxxxx",
#     modal_secret="ws-xxxxx",
# )

# Generate 3D from image
result = client.generate(
    image_path="input.png",
    seed=42,
    pipeline_type="1024_cascade",  # Options: 512, 1024, 1024_cascade, 1536_cascade
)

# result contains:
# - state: compressed state for GLB extraction
# - video: base64-encoded MP4 preview

# Extract GLB
client.extract_glb(
    state=result["state"],
    output_path="output.glb",
    decimation_target=500000,
    texture_size=2048,
)
```

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

### GLB Extraction Presets

| Preset | decimation_target | texture_size | File Size |
|--------|-------------------|--------------|-----------|
| Quality | 1,000,000 | 4096 | ~30MB |
| Balanced | 500,000 | 2048 | ~15MB |
| Fast | 100,000 | 1024 | ~5MB |

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
