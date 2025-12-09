import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image

from ..renderers import MeshRenderer, VoxelRenderer, PbrMeshRenderer
from ..representations import Mesh, Voxel, MeshWithPbrMaterial, MeshWithVoxel
from .random_utils import sphere_hammersley_sequence


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    if isinstance(sample, (MeshWithPbrMaterial, MeshWithVoxel)):
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 2)
        renderer.rendering_options.peel_layers = kwargs.get('peel_layers', 8)
    elif isinstance(sample, Mesh):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 2)
        renderer.rendering_options.chunk_size = kwargs.get('chunk_size', None)
    elif isinstance(sample, Voxel):
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.1)
        renderer.rendering_options.far = kwargs.get('far', 10.0)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 2)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), total=len(extrinsics), desc='Rendering', disable=not verbose):
        res = renderer.render(sample, extr, intr, **kwargs)
        for k, v in res.items():
            if k not in rets: rets[k] = []
            if v.dim() == 2: v = v[None].repeat(3, 1, 1)
            rets[k].append(np.clip(v.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
    return rets


def render_video(sample, resolution=1024, bg_color=(0, 0, 0), num_frames=120, r=2, fov=40, **kwargs):
    yaws = -torch.linspace(0, 2 * 3.1415, num_frames) + np.pi/2
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, nviews=4, **kwargs):
    yaw = np.linspace(0, 2 * np.pi, nviews, endpoint=False)
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(nviews)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def make_pbr_vis_frames(result, resolution=1024):
    num_frames = len(result['shaded'])
    frames = []
    for i in range(num_frames):
        shaded = Image.fromarray(result['shaded'][i])
        normal = Image.fromarray(result['normal'][i])
        base_color = Image.fromarray(result['base_color'][i])
        metallic = Image.fromarray(result['metallic'][i])
        roughness = Image.fromarray(result['roughness'][i])
        alpha = Image.fromarray(result['alpha'][i])
        shaded = shaded.resize((resolution, resolution))
        normal = normal.resize((resolution, resolution))
        base_color = base_color.resize((resolution//2, resolution//2))
        metallic = metallic.resize((resolution//2, resolution//2))
        roughness = roughness.resize((resolution//2, resolution//2))
        alpha = alpha.resize((resolution//2, resolution//2))
        row1 = np.concatenate([shaded, normal], axis=1)
        row2 = np.concatenate([base_color, metallic, roughness, alpha], axis=1)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    return frames
