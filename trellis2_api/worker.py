import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import time
import torch
import logging
import threading
import numpy as np
import imageio
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
import cv2

from .models import TaskType, TaskStatus
from .web_utils import get_next_task, update_task_status
from .config import MODEL_ID, TEXT_TO_IMAGE_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRELLIS2-Worker")

class Worker:
    def __init__(self):
        self.pipeline = None
        self.text_to_image_pipeline = None
        self.envmap = None
        self.running = False
        
    def load_models(self):
        logger.info("Loading TRELLIS2 models...")
        
        # Load main pipeline
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
        self.pipeline.cuda()
        
        # Load environment map for rendering
        envmap_path = "assets/hdri/forest.exr"
        if os.path.exists(envmap_path):
            envmap_img = cv2.cvtColor(
                cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), 
                cv2.COLOR_BGR2RGB
            )
            self.envmap = EnvMap(torch.tensor(envmap_img, dtype=torch.float32, device='cuda'))

        # Preload rembg
        try:
            self.pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
        except Exception:
            pass
            
        logger.info("Models loaded successfully")

    def _load_t2i(self):
        if self.text_to_image_pipeline is None:
            logger.info("Loading Text-to-Image pipeline...")
            from diffusers import ZImagePipeline
            self.text_to_image_pipeline = ZImagePipeline.from_pretrained(
                TEXT_TO_IMAGE_MODEL,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )
            self.text_to_image_pipeline.to("cuda")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        logger.info("Worker thread started")

    def loop(self):
        self.load_models()
        
        while self.running:
            try:
                task = get_next_task()
                if not task:
                    time.sleep(1)
                    continue
                
                logger.info(f"Processing task {task['request_id']} ({task['task_type']})")
                try:
                    self.process_task(task)
                except Exception as e:
                    logger.error(f"Task failed: {e}", exc_info=True)
                    update_task_status(task["request_id"], TaskStatus.ERROR.value, error=str(e))
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1)

    def process_task(self, task):
        task_type = task["task_type"]
        params = task["params"]
        output_dir = task["request_output_dir"]
        
        # Prepare image
        if task_type == TaskType.TEXT_TO_3D:
            self._load_t2i()
            generator = torch.Generator("cuda").manual_seed(params["seed"])
            image = self.text_to_image_pipeline(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt", ""),
                height=params.get("image_height", 1024),
                width=params.get("image_width", 1024),
                num_inference_steps=params.get("num_inference_steps", 9),
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
            # Save generated image
            image.save(os.path.join(output_dir, "generated_image.png"))
        else:
            image = Image.open(task["input_path"])

        # Preprocess
        if params.get("preprocess_image", True):
            image = self.pipeline.preprocess_image(image)
        
        # Save preview
        preview_filename = "preview.png"
        image.save(os.path.join(output_dir, preview_filename))

        # Run Pipeline
        outputs = self.pipeline.run(
            image,
            seed=params["seed"],
            sparse_structure_sampler_params={
                "steps": params.get("sparse_structure_steps", 12),
                "guidance_strength": params.get("sparse_structure_guidance_strength", 7.5),
                "guidance_rescale": params.get("sparse_structure_guidance_rescale", 0.7),
                "rescale_t": params.get("sparse_structure_rescale_t", 5.0),
            },
            shape_slat_sampler_params={
                "steps": params.get("shape_slat_steps", 12),
                "guidance_strength": params.get("shape_slat_guidance_strength", 7.5),
                "guidance_rescale": params.get("shape_slat_guidance_rescale", 0.5),
                "rescale_t": params.get("shape_slat_rescale_t", 3.0),
            },
            tex_slat_sampler_params={
                "steps": params.get("tex_slat_steps", 12),
                "guidance_strength": params.get("tex_slat_guidance_strength", 1.0),
                "guidance_rescale": params.get("tex_slat_guidance_rescale", 0.0),
                "rescale_t": params.get("tex_slat_rescale_t", 3.0),
            },
            preprocess_image=False,
            pipeline_type=params.get("pipeline_type", "512"),
        )[0]

        # Simplify
        outputs.simplify(16777216)

        output_files = {"preview": preview_filename}

        # Video
        if params.get("generate_video", True):
            video_frames = render_utils.make_pbr_vis_frames(
                render_utils.render_video(outputs, envmap=self.envmap) 
                if self.envmap else render_utils.render_video(outputs)
            )
            video_filename = "output.mp4"
            imageio.mimsave(os.path.join(output_dir, video_filename), video_frames, fps=15)
            output_files["video"] = video_filename

        # Model
        if params.get("generate_model", True):
            glb = o_voxel.postprocess.to_glb(
                vertices=outputs.vertices,
                faces=outputs.faces,
                attr_volume=outputs.attrs,
                coords=outputs.coords,
                attr_layout=outputs.layout,
                voxel_size=outputs.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=params.get("decimation_target", 1000000),
                texture_size=params.get("texture_size", 4096),
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False
            )
            model_filename = "output.glb"
            glb.export(os.path.join(output_dir, model_filename), extension_webp=True)
            output_files["model"] = model_filename

        update_task_status(task["request_id"], TaskStatus.COMPLETE.value, output_files=output_files)

# Global worker instance
worker = Worker()

def start_worker():
    worker.start()
