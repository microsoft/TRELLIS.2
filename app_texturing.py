import gradio as gr

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
import shutil
from typing import *
import torch
import numpy as np
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def shapeimage_to_tex(
    mesh_file: str,
    image: Image.Image,
    seed: int,
    resolution: str,
    texture_size: int,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    output = pipeline.run(
        mesh,
        image,
        seed=seed,
        preprocess_image=False,
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        resolution=int(resolution),
        texture_size=texture_size,
    )
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    output.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, glb_path


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Texturing a mesh with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2)
    * Upload a mesh and corresponding reference image (preferably with an alpha-masked foreground object) and click Generate to create a textured 3D asset.
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            mesh_file = gr.File(label="Upload Mesh", file_types=[".ply", ".obj", ".glb", ".gltf"], file_count="single")
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)
            
            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)
            
            generate_btn = gr.Button("Generate")
                
            with gr.Accordion(label="Advanced Settings", open=False):                
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)                

        with gr.Column(scale=10):
            glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
            download_btn = gr.DownloadButton(label="Download GLB")
                        

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        shapeimage_to_tex,
        inputs=[
            mesh_file, image_prompt, seed, resolution, texture_size,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[glb_output, download_btn],
    )
        

# Launch the Gradio app
if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    pipeline = Trellis2TexturingPipeline.from_pretrained('microsoft/TRELLIS.2-4B', config_file="texturing_pipeline.json")
    pipeline.cuda()
    
    demo.launch()
