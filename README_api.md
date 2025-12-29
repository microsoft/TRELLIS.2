# TRELLIS2 API Documentation(By Cursor)

This directory contains the implementation of TRELLIS2 for serverless deployment and as a FastAPI service.

## Contents

1. **cog.yaml** - Cog configuration for serverless deployment (e.g., Replicate)
2. **predict.py** - Stateless prediction endpoint for Cog
3. **api.py** - FastAPI wrapper for TRELLIS2
4. **sync_client_examples.py** - Python client examples

## Quick Start

### Option 1: Run with FastAPI (Recommended for Development)

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

2. **Start the API server:**
   ```bash
   python api.py
   ```
   
   Or use uvicorn directly:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

3. **Test the API:**
   
   Visit the interactive API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

4. **Run client examples:**
   ```bash
   # Python examples
   python client_examples.py
   
   # Or bash examples
   ./client_examples.sh
   ```

### Option 2: Deploy with Cog (for Production/Replicate)

1. **Install Cog:**
   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   sudo chmod +x /usr/local/bin/cog
   ```

2. **Build the container:**
   ```bash
   cog build
   ```

3. **Test locally:**
   ```bash
   cog predict -i image=@assets/example_image/T.png
   ```

4. **Deploy to Replicate:**
   ```bash
   cog push r8.im/your-username/trellis2
   ```

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "message": "Model loaded and ready",
  "model_loaded": true
}
```

### Image-to-3D Generation

```bash
POST /api/v1/image-to-3d
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | file | required | Input image file |
| seed | int | 42 | Random seed for reproducibility |
| randomize_seed | bool | false | Use random seed |
| preprocess_image | bool | true | Remove background and crop |
| generate_video | bool | true | Generate preview video |
| generate_model | bool | true | Generate GLB model |
| pipeline_type | str | "1024_cascade" | Quality preset (512, 1024, 1024_cascade, 1536_cascade) |
| sparse_structure_steps | int | 12 | Sparse structure sampling steps |
| sparse_structure_cfg | float | 7.5 | Sparse structure CFG strength |
| shape_slat_steps | int | 12 | Shape sampling steps |
| shape_slat_cfg | float | 7.5 | Shape CFG strength |
| tex_slat_steps | int | 12 | Texture sampling steps |
| tex_slat_cfg | float | 7.5 | Texture CFG strength |
| decimation_target | int | 1000000 | Target face count |
| texture_size | int | 4096 | Output texture resolution |

**Pipeline Types:**

- `512` - Fastest, lowest quality (good for quick previews)
- `1024` - Balanced quality and speed
- `1024_cascade` - Good quality (default, recommended)
- `1536_cascade` - Highest quality, slowest

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "message": "3D generation completed successfully",
  "video_url": "/api/v1/outputs/{job_id}/output.mp4",
  "model_url": "/api/v1/outputs/{job_id}/output.glb",
  "preview_image_url": "/api/v1/outputs/{job_id}/preview.png"
}
```

### Download Output Files

```bash
GET /api/v1/outputs/{job_id}/{filename}
```

Downloads the generated files (video, model, or preview image).

### Delete Job Outputs

```bash
DELETE /api/v1/outputs/{job_id}
```

Deletes all output files for a job.

## Usage Examples

### Python

```python
from client_examples import TRELLIS2Client

client = TRELLIS2Client("http://localhost:8000")

# Generate 3D from image
result = client.image_to_3d(
    image_path="assets/example_image/T.png",
    seed=42,
    pipeline_type="1024_cascade",
    generate_video=True,
    generate_model=True
)

# Download results
client.download_file(result['model_url'], "output.glb")
client.download_file(result['video_url'], "output.mp4")
```

### cURL

```bash
# Generate 3D model
curl -X POST http://localhost:8000/api/v1/image-to-3d \
  -F "image=@assets/example_image/T.png" \
  -F "seed=42" \
  -F "pipeline_type=1024_cascade" \
  -F "generate_video=true" \
  -F "generate_model=true"

# Download model
curl -o output.glb http://localhost:8000/api/v1/outputs/{job_id}/output.glb
```

### JavaScript/TypeScript

```javascript
async function generateModel(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('seed', '42');
  formData.append('pipeline_type', '1024_cascade');
  formData.append('generate_video', 'true');
  formData.append('generate_model', 'true');
  
  const response = await fetch('http://localhost:8000/api/v1/image-to-3d', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log('Job ID:', result.job_id);
  
  // Download model
  const modelBlob = await fetch(`http://localhost:8000${result.model_url}`)
    .then(r => r.blob());
  
  return modelBlob;
}
```

## Cog Predict Interface

### Parameters

The `predict.py` interface supports the following parameters:

- **image** (Path): Input image file
- **seed** (int): Random seed [default: 42]
- **randomize_seed** (bool): Randomize seed [default: False]
- **preprocess_image** (bool): Remove background [default: True]
- **return_no_background** (bool): Return preprocessed image [default: False]
- **generate_video** (bool): Generate video preview [default: True]
- **generate_model** (bool): Generate GLB model [default: True]
- **pipeline_type** (str): Quality preset [default: "1024_cascade"]
- **sparse_structure_steps** (int): Sampling steps [default: 12, range: 1-50]
- **sparse_structure_cfg** (float): CFG strength [default: 7.5, range: 0-15]
- **shape_slat_steps** (int): Sampling steps [default: 12, range: 1-50]
- **shape_slat_cfg** (float): CFG strength [default: 7.5, range: 0-15]
- **tex_slat_steps** (int): Sampling steps [default: 12, range: 1-50]
- **tex_slat_cfg** (float): CFG strength [default: 7.5, range: 0-15]
- **decimation_target** (int): Target faces [default: 1000000, range: 100000-2000000]
- **texture_size** (int): Texture resolution [default: 4096, range: 1024-8192]

### Outputs

- **no_background_image** (Path): Preprocessed image (if requested)
- **video** (Path): Rendered preview video (MP4)
- **model_file** (Path): 3D model (GLB format)

## Performance Tips

1. **For fastest results:** Use `pipeline_type="512"` with fewer steps (8-10)
2. **For best quality:** Use `pipeline_type="1536_cascade"` with more steps (20+)
3. **Balanced (recommended):** Use `pipeline_type="1024_cascade"` with default steps (12)
4. **Memory optimization:** The pipeline uses `low_vram=True` by default, which moves models between CPU/GPU as needed

## Requirements

### System Requirements
- NVIDIA GPU with CUDA support (12.4+)
- 16GB+ GPU memory recommended for 1536_cascade
- 8GB+ GPU memory for 1024_cascade
- 4GB+ GPU memory for 512/1024

### Python Dependencies
- torch==2.6.0
- torchvision==0.21.0
- flash-attn==2.7.3
- transformers
- pillow-simd
- imageio
- opencv-python
- and others (see cog.yaml or Dockerfile)

### Custom Extensions
The following custom CUDA extensions are built during setup:
- nvdiffrast (v0.4.0)
- nvdiffrec (renderutils branch)
- CuMesh
- FlexGEMM
- o-voxel (included in repo)
- utils3d

## Troubleshooting

### API Server Issues

**Problem:** `Model not loaded` error
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
nvidia-smi
```

**Problem:** Out of memory
- Use a lower `pipeline_type` (e.g., "512" instead of "1536_cascade")
- Reduce `decimation_target` and `texture_size`
- Reduce sampling steps

### Cog Build Issues

**Problem:** Build fails during extension compilation
- Ensure CUDA toolkit is properly installed
- Check `TORCH_CUDA_ARCH_LIST` matches your GPU architecture
- Try building with `--debug` flag: `cog build --debug`

**Problem:** o-voxel not found
- Make sure the `o-voxel` directory exists in the repo root
- The Dockerfile copies it to `/src/o-voxel` before installation

## Architecture

### Pipeline Flow

```
Input Image
    ↓
Preprocessing (Background Removal)
    ↓
Image Conditioning (DINOv2/CLIP)
    ↓
Sparse Structure Generation (Flow Matching)
    ↓
Shape SLat Generation (Flow Matching)
    ↓
Texture SLat Generation (Flow Matching)
    ↓
Decode to Mesh with Voxel Attributes
    ↓
Export to GLB + Render Video
```

### Key Components

1. **Image Conditioning**: Extracts multi-scale features from input
2. **Sparse Structure Flow**: Generates 3D occupancy grid
3. **Shape SLat Flow**: Generates geometry in structured latent space
4. **Texture SLat Flow**: Generates PBR material attributes
5. **Decoder**: Converts latents to mesh using o-voxel format
6. **Renderer**: PBR rendering with environment maps

## License

This implementation follows the TRELLIS2 license. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{xiang2024trellis2,
  title={TRELLIS2: Empowering 3D Generation with Textured Voxels},
  author={Xiang, Jeffrey and others},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For issues and questions:
- TRELLIS2 Issues: https://github.com/microsoft/TRELLIS.2/issues
- Cog Documentation: https://github.com/replicate/cog
- FastAPI Documentation: https://fastapi.tiangolo.com/

