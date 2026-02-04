"""
Multi-image to 3D generation using TRELLIS 2.

This example demonstrates multi-image conditioning with two fusion modes:
- 'stochastic': Cycles through images at each step (memory efficient)
- 'multidiffusion': Averages all images at each step (higher quality)
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# Configuration
IMAGE_PATHS = [
    "assets/example_image/T.png",
    "assets/example_image/T.png",  # Add more image paths here
]
FUSION_MODE = 'multidiffusion'  # 'stochastic' or 'multidiffusion'
RESOLUTION = '1024'  # '512', '1024', or '1536'

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Multiple Images
images = [Image.open(path) for path in IMAGE_PATHS]

# 4. Extract Conditioning Features
torch.manual_seed(42)
cond_list_512 = [pipeline.get_cond([img], 512)['cond'] for img in images]
stacked_cond_512 = torch.cat(cond_list_512, dim=0)
cond_512 = {
    'cond': stacked_cond_512,
    'neg_cond': torch.zeros_like(stacked_cond_512[:1])
}

cond_1024 = None
if RESOLUTION != '512':
    cond_list_1024 = [pipeline.get_cond([img], 1024)['cond'] for img in images]
    stacked_cond_1024 = torch.cat(cond_list_1024, dim=0)
    cond_1024 = {
        'cond': stacked_cond_1024,
        'neg_cond': torch.zeros_like(stacked_cond_1024[:1])
    }

# 5. Sample with Multi-Image Conditioning
pipeline_type = {'512': '512', '1024': '1024_cascade', '1536': '1536_cascade'}[RESOLUTION]
ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
ss_steps = pipeline.sparse_structure_sampler_params.get('steps', 12)
shape_steps = pipeline.shape_slat_sampler_params.get('steps', 12)
tex_steps = pipeline.tex_slat_sampler_params.get('steps', 12)

# Sample sparse structure
with pipeline.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=FUSION_MODE):
    coords = pipeline.sample_sparse_structure(cond_512, ss_res, num_samples=1, sampler_params={})

# Sample shape latent
if pipeline_type == '512':
    with pipeline.inject_sampler_multi_image('shape_slat_sampler', len(images), shape_steps, mode=FUSION_MODE):
        shape_slat = pipeline.sample_shape_slat(cond_512, pipeline.models['shape_slat_flow_model_512'], coords, {})
    tex_cond = cond_512
    tex_model = pipeline.models['tex_slat_flow_model_512']
    res = 512
elif pipeline_type == '1024':
    with pipeline.inject_sampler_multi_image('shape_slat_sampler', len(images), shape_steps, mode=FUSION_MODE):
        shape_slat = pipeline.sample_shape_slat(cond_1024, pipeline.models['shape_slat_flow_model_1024'], coords, {})
    tex_cond = cond_1024
    tex_model = pipeline.models['tex_slat_flow_model_1024']
    res = 1024
else:  # 1024_cascade or 1536_cascade
    target_res = 1024 if pipeline_type == '1024_cascade' else 1536
    with pipeline.inject_sampler_multi_image('shape_slat_sampler', len(images), shape_steps, mode=FUSION_MODE):
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models['shape_slat_flow_model_512'],
            pipeline.models['shape_slat_flow_model_1024'],
            512, target_res, coords, {}, max_num_tokens=49152
        )
    tex_cond = cond_1024
    tex_model = pipeline.models['tex_slat_flow_model_1024']

# Sample texture latent
with pipeline.inject_sampler_multi_image('tex_slat_sampler', len(images), tex_steps, mode=FUSION_MODE):
    tex_slat = pipeline.sample_tex_slat(tex_cond, tex_model, shape_slat, {})

# 6. Decode to Mesh
meshes = pipeline.decode_latent(shape_slat, tex_slat, res)
mesh = meshes[0]
mesh.simplify(16777216)  # nvdiffrast limit

# 7. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample_multi.mp4", video, fps=15)

# 8. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample_multi.glb", extension_webp=True)
