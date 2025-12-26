# TRELLIS.2 Modal Operations Runbook

This runbook covers monitoring, troubleshooting, and operational procedures for the TRELLIS.2 Modal deployment.

## Monitoring

### Health Check

The `/health` endpoint provides basic liveness status:

```bash
curl https://your-app--health.modal.run
```

Expected response:
```json
{"status": "ok", "service": "trellis2-api"}
```

### Detailed Health (via Modal method)

For detailed diagnostics, use the `health_check` method:

```python
# In Modal shell or via remote call
service = TRELLIS2Service()
result = service.health_check.remote()
print(result)
```

Response includes:
- `status`: "healthy" or "unhealthy"
- `gpu`: GPU device name
- `vram_allocated_gb`: Current VRAM usage
- `vram_total_gb`: Total VRAM
- `load_time_seconds`: Model load time

### Modal Dashboard

Monitor at: https://modal.com/apps

Key metrics:
- Container count and utilization
- Request latency (P50, P95, P99)
- Error rate
- Cold start frequency

## Common Issues

### 1. CUDA Out of Memory (OOM)

**Symptoms:**
```json
{"error": {"code": "cuda_oom", "message": "GPU out of memory..."}}
```

**Causes:**
- High-resolution pipeline (1536_cascade) with complex image
- Large texture size during GLB extraction
- Memory fragmentation from repeated requests

**Solutions:**
1. Use lower resolution pipeline (`1024` instead of `1536_cascade`)
2. Reduce texture size (2048 instead of 4096)
3. The service automatically calls `torch.cuda.empty_cache()` after operations
4. If persistent, restart the container (Modal will auto-restart)

### 2. Slow Cold Starts

**Symptoms:**
- First request takes 2-2.5 minutes

**Causes:**
- Container starting from cold (no warm instances)
- Model loading (~124s): TRELLIS.2 model + DINOv3 + RMBG-2.0
- GPU initialization for flex_gemm, Triton, flash_attn

**Solutions:**
1. Always use `modal deploy` for production
2. Use `scaledown_window=300` to keep containers warm (5 min idle timeout)
3. Implement a warm-up cron job that pings every 4 minutes:
   ```bash
   # Example cron: */4 * * * * curl -s https://your-app--health.modal.run
   ```

**Note on Memory Snapshots:**

GPU Memory Snapshots were tested (2025-12-21) and found **not effective**:
- Baseline cold start: ~143s
- With GPU snapshots: ~146s (no improvement)

Root cause: flex_gemm, Triton, and flash_attn require re-initialization after
snapshot restoration, negating any benefits from preserved model weights.

### 3. Authentication Failed

**Symptoms:**
- HTTP 401 Unauthorized response
- Request rejected before reaching endpoint

**Causes:**
- Missing `Modal-Key` or `Modal-Secret` headers
- Invalid or expired Proxy Auth Token
- Token doesn't have access to this app

**Solutions:**
1. Verify headers are set correctly:
   ```bash
   curl -X POST https://your-app--generate.modal.run \
     -H "Modal-Key: wk-xxxxx" \
     -H "Modal-Secret: ws-xxxxx" \
     -H "Content-Type: application/json" \
     -d '{"image": "...", "seed": 42}'
   ```
2. Create new token at https://modal.com/settings/proxy-auth-tokens
3. Update client credentials:
   - Environment variables: `TRELLIS2_MODAL_KEY` and `TRELLIS2_MODAL_SECRET`
   - Or secrets file: `~/.trellis2_modal_secrets.json`

### 4. Image Too Large

**Symptoms:**
```json
{"error": {"code": "validation_error", "message": "Image size exceeds limit..."}}
```

**Causes:**
- Input image exceeds 10MB (base64 decoded size)
- Image dimensions exceed 4096x4096

**Solutions:**
1. Resize image before sending
2. Use JPEG instead of PNG for smaller size
3. The pipeline will preprocess/resize automatically, but the upload must be within limits

### 5. Connection Timeouts

**Symptoms:**
- Client throws `requests.exceptions.Timeout`
- Requests take >10 minutes

**Causes:**
- Cold start + generation time exceeds timeout
- Modal infrastructure issues

**Solutions:**
1. Client has 10-minute timeout by default
2. For 1536_cascade, total time can exceed 2 minutes
3. Check Modal status page for outages

### 6. GLB Extraction Fails

**Symptoms:**
```json
{"error": {"code": "extraction_error", "message": "GLB extraction failed..."}}
```

**Causes:**
- Invalid state data
- State from incompatible version
- VRAM exhaustion during remeshing

**Solutions:**
1. Regenerate the 3D model
2. Try smaller decimation_target
3. Disable remesh: `remesh=false`

## Operational Procedures

### Deploying Updates

```bash
# 1. Run tests locally
pytest trellis2_modal/tests/

# 2. Verify image builds
modal run trellis2_modal/service/image.py

# 3. Deploy
modal deploy -m trellis2_modal.service.service

# 4. Verify health
curl https://your-app--health.modal.run
```

### Rolling Back

Modal keeps previous deployments. To rollback:

1. Go to Modal dashboard → Apps → Your app
2. Find previous deployment in history
3. Click "Redeploy" on that version

### Scaling

Modal automatically scales based on request load. To adjust:

```python
# In service.py, modify @app.cls parameters:
@app.cls(
    ...
    concurrency_limit=5,      # Max concurrent requests per container
    container_idle_timeout=300,  # Seconds before scaling down
    ...
)
```

### Managing Authentication

Authentication uses Modal Proxy Auth Tokens. Manage tokens in the Modal dashboard:

1. **Create new token**: https://modal.com/settings/proxy-auth-tokens → "New Token"
2. **Revoke token**: Click the token in the dashboard → "Delete"

Client credentials are stored locally (not in the repository):
- Environment variables: `TRELLIS2_MODAL_KEY` and `TRELLIS2_MODAL_SECRET`
- Or secrets file: `~/.trellis2_modal_secrets.json`

```json
{
    "modal_key": "wk-xxxxx",
    "modal_secret": "ws-xxxxx"
}
```

### Viewing Logs

```bash
# View recent logs
modal app logs trellis2-3d

# Stream logs
modal app logs trellis2-3d --follow
```

### Checking Volume Contents

```bash
# List files in HuggingFace cache volume
modal volume ls trellis2-hf-cache /cache/huggingface/
```

## Performance Benchmarks

Expected times on A100-80GB (warm container):

| Operation | Time |
|-----------|------|
| Generate (512) | 8-12s |
| Generate (1024) | 20-30s |
| Generate (1024_cascade) | 25-35s |
| Generate (1536_cascade) | 80-100s |
| Extract GLB (quality) | 30-60s |
| Extract GLB (fast) | 10-20s |
| Video render | 5-10s |

Cold start adds 90-120 seconds.

## Alerts to Set Up

Recommended Modal/external monitoring alerts:

1. **Error rate > 5%** - Indicates systematic issue
2. **P95 latency > 5 minutes** - Cold starts or overload
3. **OOM errors > 10/hour** - Memory pressure
4. **Health check failures** - Service down

## Emergency Procedures

### Service Completely Down

1. Check Modal status: https://status.modal.com/
2. Try redeploying: `modal deploy -m trellis2_modal.service.service`
3. Check logs for errors: `modal app logs trellis2-3d`
4. If HuggingFace is down, cached models may still work

### Data Recovery

Model weights are cached in the `trellis2-hf-cache` volume.

Authentication uses Modal Proxy Auth Tokens (managed in Modal dashboard), so there's no
authentication data to backup from volumes.
