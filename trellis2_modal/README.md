# TRELLIS.2 on Modal

Deploy TRELLIS.2 to Modal's serverless GPU infrastructure. Generate 3D assets from images without needing a local GPU.

**Why use this?**
- No local GPU required - runs on cloud A100s
- Pay only for what you use (~$0.05 per generation)
- Access via web UI or API from any device

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

## API Usage

```python
from trellis2_modal.client import TRELLIS2APIClient

client = TRELLIS2APIClient(
    base_url="https://yourname--trellis2-3d-trellis2service-generate.modal.run"
)

# Generate 3D from image
result = client.generate(image_path="input.png", pipeline_type="1024_cascade")

# Extract GLB mesh
client.extract_glb(state=result["state"], output_path="output.glb")
```

Credentials are loaded automatically from `~/.trellis2_modal_secrets.json` or environment variables.

## Pipeline Options

| Pipeline | Resolution | Time | Use Case |
|----------|------------|------|----------|
| `512` | 512³ | ~10s | Quick preview |
| `1024_cascade` | 1024³ | ~30s | Recommended |
| `1536_cascade` | 1536³ | ~90s | Maximum quality |

## Cost

| What | GPU Time | Cost |
|------|----------|------|
| Cold start | 2-3 min | ~$0.10 |
| Generation (1024) | 30s | ~$0.02 |
| GLB extraction | 30-60s | ~$0.03 |

Containers stay warm for 5 minutes between requests (no cold start cost for sequential generations).

## Troubleshooting

**403 from HuggingFace**: Accept all three model licenses (links above).

**Cold start every time**: Containers scale to zero after 5 minutes idle. Use a cron job to ping the health endpoint every 4 minutes to keep warm.

**CUDA OOM**: Use a lower resolution pipeline or reduce GLB texture_size.

## Full Documentation

See [docs/MODAL_INTEGRATION.md](docs/MODAL_INTEGRATION.md) for:
- Detailed configuration options
- Operations runbook
- Cold start optimization strategies
- Complete API reference
