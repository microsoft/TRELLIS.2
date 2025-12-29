"""
FastAPI wrapper for TRELLIS2
Provides RESTful API endpoints for text-to-3d and image-to-3d generation
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import cv2
import torch
import numpy as np
import imageio
from PIL import Image
import shutil
from pathlib import Path as PathLib
import logging
import uuid

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TRELLIS2 API",
    description="API for TRELLIS2 3D generation from images and text",
    version="1.0.0"
)

# Global variables for models
pipeline = None
text_to_image_pipeline = None
envmap = None
output_dir = PathLib("outputs")
output_dir.mkdir(exist_ok=True)

MAX_SEED = np.iinfo(np.int32).max


# Request/Response Models
class ImageTo3DRequest(BaseModel):
    """Request model for image-to-3D generation"""
    seed: int = Field(default=42, description="Random seed for generation")
    randomize_seed: bool = Field(default=False, description="Randomize seed")
    preprocess_image: bool = Field(default=True, description="Preprocess image (remove background)")
    generate_video: bool = Field(default=True, description="Generate rendered video")
    generate_model: bool = Field(default=True, description="Generate GLB model")
    pipeline_type: str = Field(
        default="512", 
        description="Pipeline type",
        pattern="^(512|1024|1024_cascade|1536_cascade)$"
    )
    sparse_structure_steps: int = Field(default=12, ge=1, le=50)
    sparse_structure_guidance_strength: float = Field(default=7.5, ge=0.0, le=15.0)
    sparse_structure_guidance_rescale: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_structure_rescale_t: float = Field(default=5.0, ge=1.0, le=6.0)
    shape_slat_steps: int = Field(default=12, ge=1, le=50)
    shape_slat_guidance_strength: float = Field(default=7.5, ge=0.0, le=15.0)
    shape_slat_guidance_rescale: float = Field(default=0.5, ge=0.0, le=1.0)
    shape_slat_rescale_t: float = Field(default=3.0, ge=1.0, le=6.0)
    tex_slat_steps: int = Field(default=12, ge=1, le=50)
    tex_slat_guidance_strength: float = Field(default=1.0, ge=0.0, le=15.0)
    tex_slat_guidance_rescale: float = Field(default=0.0, ge=0.0, le=1.0)
    tex_slat_rescale_t: float = Field(default=3.0, ge=1.0, le=6.0)
    decimation_target: int = Field(default=1000000, ge=100000, le=2000000)
    texture_size: int = Field(default=4096, ge=1024, le=8192)


class GenerationResponse(BaseModel):
    """Response model for generation requests"""
    job_id: str
    status: str
    message: str
    video_url: Optional[str] = None
    model_url: Optional[str] = None
    preview_image_url: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    model_loaded: bool
    text_to_image_loaded: bool = False


@app.on_event("startup")
async def startup_event():
    """Initialize the models on startup"""
    global pipeline, text_to_image_pipeline, envmap
    
    logger.info("Starting TRELLIS2 API server...")
    logger.info("Loading TRELLIS2 pipeline...")
    
    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.cuda()
        logger.info("TRELLIS2 pipeline loaded successfully!")
        
        # Load text-to-image pipeline (lazy loading)
        logger.info("Text-to-image pipeline will be loaded on first use...")
        text_to_image_pipeline = None
        
        # Load environment map
        envmap_path = "assets/hdri/forest.exr"
        if os.path.exists(envmap_path):
            envmap_img = cv2.cvtColor(
                cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), 
                cv2.COLOR_BGR2RGB
            )
            envmap = EnvMap(torch.tensor(envmap_img, dtype=torch.float32, device='cuda'))
            logger.info("Environment map loaded!")
        else:
            logger.warning("Environment map not found, proceeding without it")
            envmap = None
        
        # Preload rembg
        logger.info("Preloading rembg...")
        try:
            pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
        except Exception as e:
            logger.warning(f"Rembg preload warning: {str(e)}")
        
        logger.info("Startup complete!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def load_text_to_image_pipeline():
    """Lazy load text-to-image pipeline (cached after first load)"""
    global text_to_image_pipeline
    
    if text_to_image_pipeline is None:
        logger.info("Loading Z-Image text-to-image pipeline for the first time (this may take a while)...")
        try:
            from diffusers import ZImagePipeline
            text_to_image_pipeline = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )
            text_to_image_pipeline.to("cuda")
            logger.info("✓ Text-to-image pipeline loaded and cached successfully!")
        except Exception as e:
            logger.error(f"Failed to load text-to-image pipeline: {str(e)}")
            raise
    else:
        logger.info("✓ Using cached text-to-image pipeline (already loaded)")
    
    return text_to_image_pipeline


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="ok",
        message="TRELLIS2 API is running",
        model_loaded=pipeline is not None,
        text_to_image_loaded=text_to_image_pipeline is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    trellis_loaded = pipeline is not None
    t2i_loaded = text_to_image_pipeline is not None
    
    if trellis_loaded and t2i_loaded:
        status = "healthy"
        message = "All models loaded and ready"
    elif trellis_loaded:
        status = "healthy"
        message = "TRELLIS2 loaded (text-to-image will load on first use)"
    else:
        status = "unhealthy"
        message = "TRELLIS2 model not loaded"
    
    return HealthResponse(
        status=status,
        message=message,
        model_loaded=trellis_loaded,
        text_to_image_loaded=t2i_loaded
    )


@app.post("/api/v1/image-to-3d", response_model=GenerationResponse)
async def image_to_3d(
    image: UploadFile = File(..., description="Input image file"),
    seed: int = Form(42),
    randomize_seed: bool = Form(False),
    preprocess_image: bool = Form(True),
    generate_video: bool = Form(True),
    generate_model: bool = Form(True),
    pipeline_type: str = Form("512"),
    sparse_structure_steps: int = Form(12),
    sparse_structure_guidance_strength: float = Form(7.5),
    sparse_structure_guidance_rescale: float = Form(0.7),
    sparse_structure_rescale_t: float = Form(5.0),
    shape_slat_steps: int = Form(12),
    shape_slat_guidance_strength: float = Form(7.5),
    shape_slat_guidance_rescale: float = Form(0.5),
    shape_slat_rescale_t: float = Form(3.0),
    tex_slat_steps: int = Form(12),
    tex_slat_guidance_strength: float = Form(1.0),
    tex_slat_guidance_rescale: float = Form(0.0),
    tex_slat_rescale_t: float = Form(3.0),
    decimation_target: int = Form(1000000),
    texture_size: int = Form(4096),
):
    """
    Generate 3D model from input image
    
    Args:
        image: Input image file (PNG, JPG, etc.)
        seed: Random seed for reproducibility
        randomize_seed: Whether to use a random seed
        preprocess_image: Whether to remove background and crop
        generate_video: Whether to generate preview video
        generate_model: Whether to generate GLB model
        pipeline_type: Quality/speed tradeoff (512, 1024, 1024_cascade, 1536_cascade)
        sparse_structure_steps: Sampling steps for sparse structure
        sparse_structure_guidance_strength: Guidance strength for sparse structure
        sparse_structure_guidance_rescale: Guidance rescale for sparse structure
        sparse_structure_rescale_t: Rescale T for sparse structure
        shape_slat_steps: Sampling steps for shape
        shape_slat_guidance_strength: Guidance strength for shape
        shape_slat_guidance_rescale: Guidance rescale for shape
        shape_slat_rescale_t: Rescale T for shape
        tex_slat_steps: Sampling steps for texture
        tex_slat_guidance_strength: Guidance strength for texture
        tex_slat_guidance_rescale: Guidance rescale for texture
        tex_slat_rescale_t: Rescale T for texture
        decimation_target: Target face count for decimation
        texture_size: Output texture resolution
    
    Returns:
        GenerationResponse with URLs to generated assets
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    job_dir = output_dir / job_id
    job_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting job {job_id}")
    
    try:
        # Save uploaded image
        input_path = job_dir / "input.png"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Load image
        input_image = Image.open(input_path)
        
        # Preprocess if requested
        if preprocess_image:
            logger.info(f"[{job_id}] Preprocessing image...")
            processed_image = pipeline.preprocess_image(input_image)
            
            # Save preprocessed image
            preview_path = job_dir / "preview.png"
            processed_image.save(str(preview_path))
        else:
            processed_image = input_image
            preview_path = input_path
        
        # Set seed
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
        logger.info(f"[{job_id}] Using seed: {seed}")
        
        # Validate pipeline type
        if pipeline_type not in ["512", "1024", "1024_cascade", "1536_cascade"]:
            raise HTTPException(status_code=400, detail=f"Invalid pipeline_type: {pipeline_type}")
        
        # Run pipeline
        logger.info(f"[{job_id}] Running TRELLIS2 pipeline...")
        mesh = pipeline.run(
            processed_image,
            seed=seed,
            sparse_structure_sampler_params={
                "steps": sparse_structure_steps,
                "guidance_strength": sparse_structure_guidance_strength,
                "guidance_rescale": sparse_structure_guidance_rescale,
                "rescale_t": sparse_structure_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            preprocess_image=False,
            pipeline_type=pipeline_type,
        )[0]
        
        logger.info(f"[{job_id}] Pipeline complete, simplifying mesh...")
        mesh.simplify(16777216)  # nvdiffrast limit
        
        video_url = None
        model_url = None
        
        # Generate video if requested
        if generate_video:
            logger.info(f"[{job_id}] Rendering video...")
            if envmap is not None:
                video_frames = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh, envmap=envmap)
                )
            else:
                video_frames = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh)
                )
            
            video_path = job_dir / "output.mp4"
            imageio.mimsave(str(video_path), video_frames, fps=15)
            video_url = f"/api/v1/outputs/{job_id}/output.mp4"
            logger.info(f"[{job_id}] Video saved")
        
        # Generate GLB if requested
        if generate_model:
            logger.info(f"[{job_id}] Generating GLB...")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False
            )
            model_path = job_dir / "output.glb"
            glb.export(str(model_path), extension_webp=True)
            model_url = f"/api/v1/outputs/{job_id}/output.glb"
            logger.info(f"[{job_id}] GLB saved")
        
        logger.info(f"[{job_id}] Job complete!")
        
        return GenerationResponse(
            job_id=job_id,
            status="completed",
            message="3D generation completed successfully",
            video_url=video_url,
            model_url=model_url,
            preview_image_url=f"/api/v1/outputs/{job_id}/preview.png"
        )
        
    except Exception as e:
        logger.error(f"[{job_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/api/v1/outputs/{job_id}/{filename}")
async def get_output_file(job_id: str, filename: str):
    """
    Download generated output files
    
    Args:
        job_id: The job ID
        filename: The filename to download
    
    Returns:
        File response with the requested file
    """
    file_path = output_dir / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    media_types = {
        ".mp4": "video/mp4",
        ".glb": "model/gltf-binary",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    
    suffix = file_path.suffix.lower()
    media_type = media_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )


@app.post("/api/v1/text-to-3d", response_model=GenerationResponse)
async def text_to_3d(
    prompt: str = Form(..., description="Text prompt for image generation"),
    negative_prompt: str = Form("", description="Negative prompt for image generation"),
    image_width: int = Form(1024, ge=512, le=2048, description="Generated image width"),
    image_height: int = Form(1024, ge=512, le=2048, description="Generated image height"),
    num_inference_steps: int = Form(9, ge=4, le=50, description="Number of inference steps for image generation"),
    seed: int = Form(42),
    randomize_seed: bool = Form(False),
    preprocess_image: bool = Form(True),
    generate_video: bool = Form(True),
    generate_model: bool = Form(True),
    pipeline_type: str = Form("512"),
    sparse_structure_steps: int = Form(12),
    sparse_structure_guidance_strength: float = Form(7.5),
    sparse_structure_guidance_rescale: float = Form(0.7),
    sparse_structure_rescale_t: float = Form(5.0),
    shape_slat_steps: int = Form(12),
    shape_slat_guidance_strength: float = Form(7.5),
    shape_slat_guidance_rescale: float = Form(0.5),
    shape_slat_rescale_t: float = Form(3.0),
    tex_slat_steps: int = Form(12),
    tex_slat_guidance_strength: float = Form(1.0),
    tex_slat_guidance_rescale: float = Form(0.0),
    tex_slat_rescale_t: float = Form(3.0),
    decimation_target: int = Form(1000000),
    texture_size: int = Form(4096),
):
    """
    Generate 3D model from text prompt
    
    First generates an image from the text prompt using Z-Image,
    then converts it to 3D using TRELLIS2.
    
    Args:
        prompt: Text description of the desired image/3D model
        negative_prompt: Things to avoid in the generation
        image_width: Width of generated image
        image_height: Height of generated image
        num_inference_steps: Steps for image generation (9 recommended for Turbo)
        seed: Random seed for reproducibility
        randomize_seed: Whether to use a random seed
        preprocess_image: Whether to remove background and crop
        generate_video: Whether to generate preview video
        generate_model: Whether to generate GLB model
        pipeline_type: Quality/speed tradeoff (512, 1024, 1024_cascade, 1536_cascade)
        ... (other 3D generation parameters same as image-to-3d)
    
    Returns:
        GenerationResponse with URLs to generated assets
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TRELLIS2 model not loaded")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    job_dir = output_dir / job_id
    job_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting text-to-3d job {job_id}")
    
    try:
        # Step 1: Generate image from text
        logger.info(f"[{job_id}] Generating image from text prompt...")
        
        # Load text-to-image pipeline
        t2i_pipe = load_text_to_image_pipeline()
        
        # Set seed
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
        logger.info(f"[{job_id}] Using seed: {seed}")
        
        # Generate image
        generator = torch.Generator("cuda").manual_seed(seed)
        generated_image = t2i_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=image_height,
            width=image_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # Guidance should be 0 for Turbo models
            generator=generator,
        ).images[0]
        
        # Save generated image
        input_path = job_dir / "generated_image.png"
        generated_image.save(str(input_path))
        logger.info(f"[{job_id}] Image generated and saved")
        
        # Step 2: Preprocess if requested
        if preprocess_image:
            logger.info(f"[{job_id}] Preprocessing image...")
            processed_image = pipeline.preprocess_image(generated_image)
            
            # Save preprocessed image
            preview_path = job_dir / "preview.png"
            processed_image.save(str(preview_path))
        else:
            processed_image = generated_image
            preview_path = input_path
        
        # Validate pipeline type
        if pipeline_type not in ["512", "1024", "1024_cascade", "1536_cascade"]:
            raise HTTPException(status_code=400, detail=f"Invalid pipeline_type: {pipeline_type}")
        
        # Step 3: Run TRELLIS2 pipeline
        logger.info(f"[{job_id}] Running TRELLIS2 pipeline...")
        mesh = pipeline.run(
            processed_image,
            seed=seed,
            sparse_structure_sampler_params={
                "steps": sparse_structure_steps,
                "guidance_strength": sparse_structure_guidance_strength,
                "guidance_rescale": sparse_structure_guidance_rescale,
                "rescale_t": sparse_structure_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            preprocess_image=False,
            pipeline_type=pipeline_type,
        )[0]
        
        logger.info(f"[{job_id}] Pipeline complete, simplifying mesh...")
        mesh.simplify(16777216)  # nvdiffrast limit
        
        video_url = None
        model_url = None
        
        # Generate video if requested
        if generate_video:
            logger.info(f"[{job_id}] Rendering video...")
            if envmap is not None:
                video_frames = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh, envmap=envmap)
                )
            else:
                video_frames = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh)
                )
            
            video_path = job_dir / "output.mp4"
            imageio.mimsave(str(video_path), video_frames, fps=15)
            video_url = f"/api/v1/outputs/{job_id}/output.mp4"
            logger.info(f"[{job_id}] Video saved")
        
        # Generate GLB if requested
        if generate_model:
            logger.info(f"[{job_id}] Generating GLB...")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False
            )
            model_path = job_dir / "output.glb"
            glb.export(str(model_path), extension_webp=True)
            model_url = f"/api/v1/outputs/{job_id}/output.glb"
            logger.info(f"[{job_id}] GLB saved")
        
        logger.info(f"[{job_id}] Job complete!")
        
        return GenerationResponse(
            job_id=job_id,
            status="completed",
            message="Text-to-3D generation completed successfully",
            video_url=video_url,
            model_url=model_url,
            preview_image_url=f"/api/v1/outputs/{job_id}/preview.png"
        )
        
    except Exception as e:
        logger.error(f"[{job_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.delete("/api/v1/outputs/{job_id}")
async def delete_job_outputs(job_id: str):
    """
    Delete all outputs for a job
    
    Args:
        job_id: The job ID
    
    Returns:
        Success message
    """
    job_dir = output_dir / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        shutil.rmtree(job_dir)
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Deleted outputs for job {job_id}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete outputs: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

