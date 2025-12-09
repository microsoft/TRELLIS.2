import torch
import numpy as np
import imageio
import o_voxel
import utils3d

RES = 512

# Load data
coords, data = o_voxel.io.read("ovoxel_helmet.vxz")
position = (coords / RES - 0.5).cuda()
base_color = (data['base_color'] / 255).cuda()

# Setup camera
extr = utils3d.extrinsics_look_at(
    eye=torch.tensor([1.2, 0.5, 1.2]),
    look_at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0])
).cuda()
intr = utils3d.intrinsics_from_fov_xy(
    fov_x=torch.deg2rad(torch.tensor(45.0)),
    fov_y=torch.deg2rad(torch.tensor(45.0)),
).cuda()

# Render
renderer = o_voxel.rasterize.VoxelRenderer(
    rendering_options={"resolution": 512, "ssaa": 2}
)
output = renderer.render(
    position=position,          # Voxel centers
    attrs=base_color,           # Color/Opacity etc.
    voxel_size=1.0/RES,
    extrinsics=extr,
    intrinsics=intr
)
image = np.clip(
    output.attr.permute(1, 2, 0).cpu().numpy() * 255, 0, 255
).astype(np.uint8)
imageio.imwrite("ovoxel_helmet_visualization.png", image)
