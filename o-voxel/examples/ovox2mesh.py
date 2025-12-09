import torch
import o_voxel
import trimesh
import trimesh.visual

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

# Save as ply
visual = trimesh.visual.ColorVisuals(
    vertex_colors=base_color,
)
mesh = trimesh.Trimesh(
    vertices=rec_verts.cpu(), faces=rec_faces.cpu(), visual=visual,
    process=False
)
mesh.export("rec_helmet.ply")
