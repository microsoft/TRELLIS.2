import torch
import o_voxel

RES = 512

# Load data
coords, data = o_voxel.io.read("ovoxel_helmet.vxz")
dual_vertices = data['dual_vertices']
intersected = data['intersected']
base_color = data['base_color']
metallic = data['metallic']
roughness = data['roughness']
alpha = data['alpha']

# Depack
dual_vertices = dual_vertices / 255
intersected = torch.cat([
    intersected % 2,
    intersected // 2 % 2,
    intersected // 4 % 2,
], dim=-1).bool()

# Extract Mesh
# O-Voxel connects dual vertices to form quads, optionally splitting them 
# based on geometric features.
rec_verts, rec_faces = o_voxel.convert.flexible_dual_grid_to_mesh(
    coords.cuda(), 
    dual_vertices.cuda(), 
    intersected.cuda(), 
    split_weight=None, # Auto-split based on min angle if None
    grid_size=RES,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
)

# Post-process
attr_volume = torch.cat([base_color.cuda(), metallic.cuda(), roughness.cuda(), alpha.cuda()], dim=-1) / 255
attr_layout = {'base_color': slice(0,3), 'metallic': slice(3,4), 'roughness': slice(4,5), 'alpha': slice(5,6)}
mesh = o_voxel.postprocess.to_glb(
    vertices=rec_verts,
    faces=rec_faces,
    attr_volume=attr_volume,
    coords=coords.cuda(),
    attr_layout=attr_layout,
    grid_size=RES,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    decimation_target=100000,
    texture_size=2048,
    verbose=True,
)

# Save as glb
mesh.export("rec_helmet.glb")
