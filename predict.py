# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, BaseModel
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import cv2
import torch
import numpy as np
import imageio
from PIL import Image
from typing import Optional
from modelscope import snapshot_download as modelscope_snapshot_download
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
import logging


MAX_SEED = np.iinfo(np.int32).max


class PredictOutput(BaseModel):
    """Output model for predictions"""
    no_background_image: Path | None = None
    video: Path | None = None
    model_file: Path | None = None


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Downloading modelscope models...")
        modelscope_snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m')
        modelscope_snapshot_download('AI-ModelScope/RMBG-2.0', ignore_patterns="*.onnx")
        self.logger.info("Loading TRELLIS2 pipeline...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        self.pipeline.cuda()
        
        self.logger.info("Loading environment maps...")
        # Load default environment map
        envmap_path = "assets/hdri/forest.exr"
        if os.path.exists(envmap_path):
            envmap_img = cv2.cvtColor(
                cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), 
                cv2.COLOR_BGR2RGB
            )
            self.envmap = EnvMap(torch.tensor(envmap_img, dtype=torch.float32, device='cuda'))
        else:
            self.logger.warning("Default environment map not found, will skip environment lighting")
            self.envmap = None
        
        self.logger.info("Preloading rembg...")
        try:
            self.pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
        except Exception as e:
            self.logger.warning(f"Rembg preload warning (this is usually fine): {str(e)}")
        
        self.logger.info("Setup complete!")

    def predict(
        self,
        image: Path = Input(description="Input image to generate 3D asset from"),
        seed: int = Input(description="Random seed for generation", default=42),
        randomize_seed: bool = Input(description="Randomize seed", default=False),
        preprocess_image: bool = Input(
            description="Preprocess image (remove background and crop)", 
            default=True
        ),
        return_no_background: bool = Input(
            description="Return the preprocessed image without background", 
            default=False
        ),
        generate_video: bool = Input(
            description="Generate rendered video preview", 
            default=True
        ),
        generate_model: bool = Input(
            description="Generate 3D model file (GLB)", 
            default=True
        ),
        pipeline_type: str = Input(
            description="Pipeline type for quality/speed tradeoff",
            choices=["512", "1024", "1024_cascade", "1536_cascade"],
            default="1024_cascade"
        ),
        sparse_structure_steps: int = Input(
            description="Sparse structure sampling steps (more steps = better quality, slower)",
            default=12,
            ge=1,
            le=50
        ),
        sparse_structure_guidance_strength: float = Input(
            description="Sparse structure guidance strength",
            default=7.5,
            ge=0.0,
            le=15.0
        ),
        sparse_structure_guidance_rescale: float = Input(
            description="Sparse structure guidance rescale",
            default=0.7,
            ge=0.0,
            le=1.0
        ),
        sparse_structure_rescale_t: float = Input(
            description="Sparse structure rescale T",
            default=5.0,
            ge=1.0,
            le=6.0
        ),
        shape_slat_steps: int = Input(
            description="Shape SLat sampling steps",
            default=12,
            ge=1,
            le=50
        ),
        shape_slat_guidance_strength: float = Input(
            description="Shape SLat guidance strength",
            default=7.5,
            ge=0.0,
            le=15.0
        ),
        shape_slat_guidance_rescale: float = Input(
            description="Shape SLat guidance rescale",
            default=0.5,
            ge=0.0,
            le=1.0
        ),
        shape_slat_rescale_t: float = Input(
            description="Shape SLat rescale T",
            default=3.0,
            ge=1.0,
            le=6.0
        ),
        tex_slat_steps: int = Input(
            description="Texture SLat sampling steps",
            default=12,
            ge=1,
            le=50
        ),
        tex_slat_guidance_strength: float = Input(
            description="Texture SLat guidance strength",
            default=1.0,
            ge=0.0,
            le=15.0
        ),
        tex_slat_guidance_rescale: float = Input(
            description="Texture SLat guidance rescale",
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        tex_slat_rescale_t: float = Input(
            description="Texture SLat rescale T",
            default=3.0,
            ge=1.0,
            le=6.0
        ),
        decimation_target: int = Input(
            description="Target number of faces for decimation (only used if generate_model=True)",
            default=1000000,
            ge=100000,
            le=2000000
        ),
        texture_size: int = Input(
            description="Texture size for GLB export (only used if generate_model=True)",
            default=4096,
            ge=1024,
            le=8192
        ),
    ) -> PredictOutput:
        """Run a single prediction on the model"""
        
        # Load and preprocess image
        self.logger.info("Loading input image...")
        input_image = Image.open(str(image))
        
        no_bg_path = None
        if preprocess_image:
            self.logger.info("Preprocessing image (removing background)...")
            processed_image = self.pipeline.preprocess_image(input_image)
            
            if return_no_background:
                no_bg_path = Path("output_no_background.png")
                processed_image.save(str(no_bg_path))
                self.logger.info("Saved image without background")
        else:
            processed_image = input_image
        
        # Randomize seed if requested
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
            self.logger.info(f"Using randomized seed: {seed}")
        else:
            self.logger.info(f"Using provided seed: {seed}")
        
        # Generate 3D asset
        self.logger.info(f"Running TRELLIS2 pipeline with type '{pipeline_type}'...")
        mesh = self.pipeline.run(
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
        
        self.logger.info("TRELLIS2 pipeline complete!")
        
        # Simplify mesh for nvdiffrast limit
        self.logger.info("Simplifying mesh...")
        mesh.simplify(16777216)
        
        # Initialize output paths as None
        video_path = None
        model_path = None
        
        # Render video if requested
        if generate_video:
            self.logger.info("Rendering video preview...")
            if self.envmap is not None:
                video_frames = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh, envmap=self.envmap)
                )
            else:
                video_frames = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh)
                )
            
            video_path = Path("output_video.mp4")
            imageio.mimsave(str(video_path), video_frames, fps=15)
            self.logger.info("Video rendering complete!")
        
        # Generate GLB if requested
        if generate_model:
            self.logger.info("Generating GLB model...")
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
            model_path = Path("output.glb")
            glb.export(str(model_path), extension_webp=True)
            self.logger.info("GLB model generation complete!")
        
        self.logger.info("Prediction complete! Returning results...")
        return PredictOutput(
            no_background_image=no_bg_path,
            video=video_path,
            model_file=model_path
        )

