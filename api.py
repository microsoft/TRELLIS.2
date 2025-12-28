"""
FastAPI backend for TRELLIS.2 3D generation pipeline.

Endpoints:
- POST /generate - Submit base64 image, returns UUID
- GET /status/{uuid} - Check generation status/progress
- GET /result/{uuid} - Get base64 GLB (textured) and GLB (mesh-only) results

Output:
- glb_base64: Full GLB with PBR textures (base color, metallic, roughness)
- mesh_only_base64: GLB with geometry only (no textures, gray material)

Memory considerations for 96GB VRAM:
- Model weights: ~8GB (4B params in fp16)
- Per generation at 1024 resolution: ~20-30GB
- Safe concurrent limit: 2 jobs (conservative), 3 jobs (aggressive)
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import asyncio
import base64
import io
import uuid
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
import o_voxel


# =============================================================================
# Configuration
# =============================================================================

# Maximum concurrent GPU jobs (for 96GB VRAM, 2 is safe, 3 is aggressive)
MAX_CONCURRENT_JOBS = 2

# Job expiration time in seconds (clean up old jobs)
JOB_EXPIRATION_SECONDS = 3600  # 1 hour

# Output directory for temporary files
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Enums and Models
# =============================================================================

class JobStatus(str, Enum):
    STARTING = "starting"                # Job accepted + preprocessing
    SCULPTING = "sculpting"              # Sparse structure generation
    MESHING = "meshing"                  # Mesh geometry generation
    TEXTURING = "texturing"              # Texture/material generation
    EXTRACTING_MESH = "extracting_mesh"  # Creating mesh-only GLB
    EXTRACTING_TEXTURED = "extracting_textured"  # Creating textured GLB
    COMPLETED = "completed"
    FAILED = "error"


class SendRequest(BaseModel):
    """
    Simplified request model for /send endpoint.
    Uses numeric resolution and texture_resolution for easier integration.
    """
    image_base64: str = Field(..., description="Base64 encoded image (PNG/JPEG with optional alpha)")
    resolution: int = Field(
        default=512,
        description="Resolution: 512, 1024, or 1536"
    )
    texture_resolution: int = Field(
        default=-1,
        description="Texture resolution in pixels. -1 = use default based on resolution, or specify: 1024, 2048, 3072, 4096"
    )
    face_count: int = Field(
        default=-1,
        description="Target face count for mesh. -1 = use default based on resolution, otherwise use specified value"
    )
    seed: Optional[int] = Field(
        default=-1,
        description="Random seed. -1 = random seed, otherwise use specified value"
    )


class GenerateRequest(BaseModel):
    """Request model for /generate endpoint (advanced)"""
    image_base64: str = Field(..., description="Base64 encoded image (PNG/JPEG with optional alpha)")
    resolution: str = Field(default="1024", description="Resolution: 512, 1024, or 1536")
    seed: Optional[int] = Field(default=None, description="Random seed (None for random)")
    decimation_target: int = Field(default=500000, description="Target face count for mesh simplification")
    texture_size: int = Field(default=2048, description="Texture resolution")
    
    # Sampling steps (optional) - guidance/rescale use fixed defaults
    ss_sampling_steps: int = Field(default=12, description="Sparse structure sampling steps")
    shape_slat_sampling_steps: int = Field(default=12, description="Shape sampling steps")
    tex_slat_sampling_steps: int = Field(default=12, description="Texture sampling steps")


class GenerateResponse(BaseModel):
    """Response model for /generate and /send endpoints"""
    job_id: str
    status: JobStatus
    message: str


class StatusResponse(BaseModel):
    """Response model for /status endpoint"""
    job_id: str
    status: JobStatus
    progress: float = Field(description="Progress percentage 0-100")
    message: str
    mesh_only_ready: bool = Field(default=False, description="True when mesh-only GLB is available")
    textured_ready: bool = Field(default=False, description="True when textured GLB is available")
    mesh_only_base64: Optional[str] = Field(default=None, description="Base64 encoded mesh-only GLB (when mesh_only_ready=True)")
    glb_base64: Optional[str] = Field(default=None, description="Base64 encoded textured GLB (when textured_ready=True)")
    created_at: str
    updated_at: str


class ResultResponse(BaseModel):
    """Response model for /result endpoint"""
    job_id: str
    status: JobStatus
    glb_base64: Optional[str] = Field(default=None, description="Base64 encoded GLB with textures")
    mesh_only_base64: Optional[str] = Field(default=None, description="Base64 encoded GLB mesh without textures (geometry only)")
    error: Optional[str] = None


@dataclass
class Job:
    """Internal job tracking"""
    job_id: str
    status: JobStatus
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    request: GenerateRequest
    glb_path: Optional[str] = None
    mesh_only_path: Optional[str] = None
    mesh_only_ready: bool = False  # True when mesh-only GLB is available
    textured_ready: bool = False   # True when textured GLB is available
    error: Optional[str] = None


# =============================================================================
# Global State
# =============================================================================

# Job storage
jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()

# Semaphore for limiting concurrent GPU operations
gpu_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS + 2)

# Pipeline (loaded once at startup)
pipeline: Optional[Trellis2ImageTo3DPipeline] = None
envmap: Optional[Dict[str, EnvMap]] = None


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="TRELLIS.2 3D Generation API",
    description="API for generating 3D models from images using TRELLIS.2-4B",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def update_job(job_id: str, **kwargs):
    """Update job status"""
    with jobs_lock:
        if job_id in jobs:
            job = jobs[job_id]
            for key, value in kwargs.items():
                setattr(job, key, value)
            job.updated_at = datetime.now()


def cleanup_old_jobs():
    """Remove expired jobs"""
    now = datetime.now()
    with jobs_lock:
        expired = [
            job_id for job_id, job in jobs.items()
            if (now - job.created_at).total_seconds() > JOB_EXPIRATION_SECONDS
        ]
        for job_id in expired:
            job = jobs.pop(job_id)
            # Clean up files
            if job.glb_path and os.path.exists(job.glb_path):
                os.remove(job.glb_path)
            if job.mesh_only_path and os.path.exists(job.mesh_only_path):
                os.remove(job.mesh_only_path)


def process_job(job_id: str):
    """
    Process a generation job (runs in background thread).
    This function acquires the GPU semaphore to limit concurrent GPU usage.
    
    Pipeline stages:
    1. STARTING (5%) - Job accepted + image preprocessing
    2. SCULPTING (10-20%) - Sparse structure generation
    3. MESHING (20-40%) - Mesh geometry generation
    4. TEXTURING (40-60%) - Texture/material generation
    5. EXTRACTING_MESH (60-75%) - Create mesh-only GLB (available for download)
    6. EXTRACTING_TEXTURED (75-95%) - Create textured GLB
    7. COMPLETED (100%)
    """
    global pipeline
    
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        request = job.request
    
    try:
        # Acquire GPU semaphore (blocks if too many concurrent jobs)
        with gpu_semaphore:
            # Stage 1: Preprocessing
            update_job(job_id, status=JobStatus.STARTING, progress=5, message="Preprocessing image...")
            
            # Decode and preprocess image
            image = decode_base64_image(request.image_base64)
            image = pipeline.preprocess_image(image)
            
            # Set seed
            seed = request.seed if request.seed is not None else np.random.randint(0, 2**31)
            
            # Stage 2: Sparse Structure Generation
            update_job(job_id, status=JobStatus.SCULPTING, progress=10, 
                      message="Generating sparse structure...")
            
            # Stage 3 & 4: Shape and Texture Generation
            # Note: pipeline.run() combines these stages internally
            # We update status to MESHING, then TEXTURING
            update_job(job_id, status=JobStatus.MESHING, progress=20,
                      message="Generating mesh geometry...")
            
            # Run pipeline (generates structure, shape, and texture)
            outputs, latents = pipeline.run(
                image,
                seed=seed,
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": request.ss_sampling_steps,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.7,
                    "rescale_t": 5.0,
                },
                shape_slat_sampler_params={
                    "steps": request.shape_slat_sampling_steps,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.5,
                    "rescale_t": 3.0,
                },
                tex_slat_sampler_params={
                    "steps": request.tex_slat_sampling_steps,
                    "guidance_strength": 1.0,
                    "guidance_rescale": 0.0,
                    "rescale_t": 3.0,
                },
                pipeline_type={
                    "512": "512",
                    "1024": "1024_cascade",
                    "1536": "1536_cascade",
                }[request.resolution],
                return_latent=True,
            )
            
            update_job(job_id, status=JobStatus.TEXTURING, progress=50,
                      message="Generating textures and materials...")
            
            mesh = outputs[0]
            mesh.simplify(16777216)  # nvdiffrast limit
            
            # Stage 5: Extract Mesh-Only GLB (FIRST - so it's available earlier)
            update_job(job_id, status=JobStatus.EXTRACTING_MESH, progress=60,
                      message="Creating mesh-only GLB (geometry without textures)...")
            
            # Unpack latents for mesh export
            from trellis2.modules.sparse import SparseTensor
            shape_slat, tex_slat, res = latents
            decoded_mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
            
            # Create mesh-only GLB first (faster, available sooner)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get vertices and faces for mesh-only
            # Use a simpler export without full texture processing
            vertices_np = decoded_mesh.vertices.cpu().numpy()
            faces_np = decoded_mesh.faces.cpu().numpy()
            
            # Swap Y and Z axes for GLB compatibility (same as to_glb does)
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
            
            # Create mesh-only GLB with gray material
            gray_material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.7, 0.7, 0.7, 1.0],  # Gray color
                metallicFactor=0.0,
                roughnessFactor=0.5,
            )
            mesh_only = trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                process=False,
                visual=trimesh.visual.TextureVisuals(material=gray_material)
            )
            mesh_only_path = os.path.join(OUTPUT_DIR, f'{job_id}_{timestamp}_mesh.glb')
            mesh_only.export(mesh_only_path, file_type='glb')
            
            # Mark mesh-only as ready (can be downloaded now!)
            update_job(job_id, progress=70, message="Mesh-only GLB ready! Creating textured version...",
                      mesh_only_path=mesh_only_path, mesh_only_ready=True)
            
            # Stage 6: Extract Textured GLB
            update_job(job_id, status=JobStatus.EXTRACTING_TEXTURED, progress=75,
                      message="Creating textured GLB (baking textures)...")
            
            # Create full GLB with textures (slower due to UV unwrapping and texture baking)
            glb = o_voxel.postprocess.to_glb(
                vertices=decoded_mesh.vertices,
                faces=decoded_mesh.faces,
                attr_volume=decoded_mesh.attrs,
                coords=decoded_mesh.coords,
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=request.decimation_target,
                texture_size=request.texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=False,
            )
            
            # Save textured GLB
            glb_path = os.path.join(OUTPUT_DIR, f'{job_id}_{timestamp}.glb')
            glb.export(glb_path, extension_webp=True)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Stage 7: Completed
            update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100,
                message="Generation complete - both mesh-only and textured GLB ready",
                glb_path=glb_path,
                textured_ready=True
            )
            
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        update_job(
            job_id,
            status=JobStatus.FAILED,
            progress=0,
            message=f"Error: {str(e)}",
            error=error_msg
        )
        torch.cuda.empty_cache()


# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load pipeline on startup"""
    global pipeline, envmap
    
    print("Loading TRELLIS.2-4B pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()
    
    print("Loading environment maps...")
    envmap = {
        'forest': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
    }
    
    print(f"API ready! Max concurrent jobs: {MAX_CONCURRENT_JOBS}")


@app.get("/")
async def root():
    """Health check endpoint"""
    active_statuses = [
        JobStatus.STARTING, JobStatus.STARTING, 
        JobStatus.SCULPTING, JobStatus.MESHING, JobStatus.TEXTURING,
        JobStatus.EXTRACTING_MESH, JobStatus.EXTRACTING_TEXTURED
    ]
    return {
        "status": "ok",
        "service": "TRELLIS.2 3D Generation API",
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "active_jobs": len([j for j in jobs.values() if j.status in active_statuses])
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
    
    processing_statuses = [
        JobStatus.STARTING,
        JobStatus.SCULPTING, JobStatus.MESHING, JobStatus.TEXTURING,
        JobStatus.EXTRACTING_MESH, JobStatus.EXTRACTING_TEXTURED
    ]
    
    return {
        "status": "healthy",
        "gpu_total_memory_gb": round(gpu_memory, 2),
        "gpu_used_memory_gb": round(gpu_memory_used, 2),
        "gpu_free_memory_gb": round(gpu_memory - gpu_memory_used, 2),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "queued_jobs": len([j for j in jobs.values() if j.status == JobStatus.STARTING]),
        "active_jobs": len([j for j in jobs.values() if j.status in processing_statuses]),
        "completed_jobs": len([j for j in jobs.values() if j.status == JobStatus.COMPLETED]),
        "failed_jobs": len([j for j in jobs.values() if j.status == JobStatus.FAILED]),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Submit an image for 3D generation.
    
    Returns a job_id that can be used to check status and retrieve results.
    """
    # Validate resolution
    if request.resolution not in ["512", "1024", "1536"]:
        raise HTTPException(status_code=400, detail="Resolution must be 512, 1024, or 1536")
    
    # Validate base64 image
    try:
        image = decode_base64_image(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    # Create job
    job_id = str(uuid.uuid4())
    now = datetime.now()
    
    job = Job(
        job_id=job_id,
        status=JobStatus.STARTING,
        progress=0,
        message="Job starting",
        created_at=now,
        updated_at=now,
        request=request
    )
    
    with jobs_lock:
        jobs[job_id] = job
    
    # Submit to thread pool
    executor.submit(process_job, job_id)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_old_jobs)
    
    return GenerateResponse(
        job_id=job_id,
        status=JobStatus.STARTING,
        message="Job submitted successfully"
    )


@app.post("/send", response_model=GenerateResponse)
async def send(request: SendRequest, background_tasks: BackgroundTasks):
    """
    Simplified endpoint to submit an image for 3D generation.
    
    Parameters:
    - image_base64: Base64 encoded image
    - resolution: 512, 1024, or 1536
    - texture_resolution: Texture size in pixels (-1 for auto based on resolution, or 1024/2048/3072/4096)
    - face_count: Target faces (-1 for auto based on resolution)
    - seed: Random seed (-1 for random, or specify a positive integer for reproducibility)
    
    Default mappings when -1 is used:
    - Resolution 512:  face_count=200000,  texture_resolution=1024
    - Resolution 1024: face_count=500000,  texture_resolution=2048
    - Resolution 1536: face_count=1000000, texture_resolution=4096
    
    Returns a job_id to check status via GET /status/{job_id}
    """
    # Validate resolution
    if request.resolution not in [512, 1024, 1536]:
        raise HTTPException(status_code=400, detail="Resolution must be 512, 1024, or 1536")
    
    # Map resolution to default decimation target
    face_count_mapping = {
        512: 200000,    # Low detail, fast
        1024: 500000,   # Medium detail
        1536: 1000000,  # High detail, slow
    }
    
    # Map resolution to default texture resolution
    texture_resolution_mapping = {
        512: 1024,      # Low detail
        1024: 2048,     # Medium detail
        1536: 4096,     # High detail
    }
    
    # Use custom face_count if specified, otherwise use default from resolution
    if request.face_count > 0:
        decimation_target = request.face_count
    else:
        decimation_target = face_count_mapping[request.resolution]
    
    # Use custom texture_resolution if specified, otherwise use default from resolution
    if request.texture_resolution > 0:
        texture_size = request.texture_resolution
    else:
        texture_size = texture_resolution_mapping[request.resolution]
    
    # Validate base64 image
    try:
        image = decode_base64_image(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    # Handle seed: -1 means random (None), otherwise use specified value
    actual_seed = None if request.seed < 0 else request.seed
    
    # Convert to full GenerateRequest
    full_request = GenerateRequest(
        image_base64=request.image_base64,
        resolution=str(request.resolution),
        seed=actual_seed,
        decimation_target=decimation_target,
        texture_size=texture_size,
    )
    
    # Create job
    job_id = str(uuid.uuid4())
    now = datetime.now()
    
    job = Job(
        job_id=job_id,
        status=JobStatus.STARTING,
        progress=0,
        message="Job starting",
        created_at=now,
        updated_at=now,
        request=full_request
    )
    
    with jobs_lock:
        jobs[job_id] = job
    
    # Submit to thread pool
    executor.submit(process_job, job_id)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_old_jobs)
    
    return GenerateResponse(
        job_id=job_id,
        status=JobStatus.STARTING,
        message=f"Job submitted - Resolution: {request.resolution}, Texture: {request.texture_resolution}px, Faces: {decimation_target}"
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """
    Get the status and progress of a generation job.
    
    Status progression:
    - starting → sculpting → meshing → texturing → 
      extracting_mesh → extracting_textured → completed
    
    When mesh_only_ready=True, response includes mesh_only_base64 (geometry-only GLB).
    When textured_ready=True (completed), response includes glb_base64 (textured GLB).
    """
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Encode files to base64 when available
    mesh_only_base64 = None
    glb_base64 = None
    
    # Include mesh-only GLB when ready
    if job.mesh_only_ready and job.mesh_only_path and os.path.exists(job.mesh_only_path):
        mesh_only_base64 = encode_file_to_base64(job.mesh_only_path)
    
    # Include textured GLB when ready
    if job.textured_ready and job.glb_path and os.path.exists(job.glb_path):
        glb_base64 = encode_file_to_base64(job.glb_path)
    
    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        mesh_only_ready=job.mesh_only_ready,
        textured_ready=job.textured_ready,
        mesh_only_base64=mesh_only_base64,
        glb_base64=glb_base64,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat()
    )


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str):
    """
    Get the result of a generation job.
    
    Returns base64 encoded GLB files:
    - mesh_only_base64: Available when mesh_only_ready=True (geometry only, no textures)
    - glb_base64: Available when textured_ready=True (full PBR textures)
    
    You can fetch mesh_only as soon as it's ready, even before texturing completes.
    """
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status == JobStatus.FAILED:
        return ResultResponse(
            job_id=job.job_id,
            status=job.status,
            error=job.error
        )
    
    # Allow partial results - mesh_only can be fetched before textured is ready
    if not job.mesh_only_ready and job.status != JobStatus.COMPLETED:
        return ResultResponse(
            job_id=job.job_id,
            status=job.status,
            error=f"No results available yet. Current status: {job.status}, progress: {job.progress}%"
        )
    
    # Encode available files to base64
    glb_base64 = None
    mesh_only_base64 = None
    
    # Textured GLB (only if ready)
    if job.textured_ready and job.glb_path and os.path.exists(job.glb_path):
        glb_base64 = encode_file_to_base64(job.glb_path)
    
    # Mesh-only GLB (available earlier)
    if job.mesh_only_ready and job.mesh_only_path and os.path.exists(job.mesh_only_path):
        mesh_only_base64 = encode_file_to_base64(job.mesh_only_path)
    
    return ResultResponse(
        job_id=job.job_id,
        status=job.status,
        glb_base64=glb_base64,
        mesh_only_base64=mesh_only_base64
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files.
    """
    with jobs_lock:
        job = jobs.pop(job_id, None)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Clean up files
    if job.glb_path and os.path.exists(job.glb_path):
        os.remove(job.glb_path)
    if job.mesh_only_path and os.path.exists(job.mesh_only_path):
        os.remove(job.mesh_only_path)
    
    return {"message": f"Job {job_id} deleted successfully"}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

