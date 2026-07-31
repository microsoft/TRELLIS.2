from typing import *
import torch
import torch.nn.functional as F_grid
import numpy as np
import os
from ..voxel import Voxel

from trellis2.backends import (
    MeshBackend as _MeshBackendClass,
    HAS_MESH as _HAS_MESH,
    HAS_MPS as _HAS_MPS,
    HAS_CUDA as _HAS_CUDA,
    HAS_TRIMESH,
    trimesh as _trimesh,
    HAS_FAST_SIMPLIFICATION,
    fast_simplification,
    HAS_FLEX_GEMM,
    _flex_grid_sample_3d as grid_sample_3d,
)


class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs

    @property
    def device(self):
        return self.vertices.device

    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )

    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)

    def cpu(self):
        return self.to('cpu')

    def _gpu_device(self):
        # Decoder-sized meshes have historically triggered watchdogs or
        # process-level crashes in Metal cleanup kernels. Keep the supported
        # default on the safe CPU implementations; advanced users may opt in.
        if _HAS_MPS and os.environ.get('TRELLIS_ENABLE_MPS_DECODE_MESH_OPS') == '1':
            return 'mps'
        if _HAS_CUDA:
            return 'cuda'
        return None

    def fill_holes(self, max_hole_perimeter=3e-2):
        if _HAS_MESH and self._gpu_device():
            dev = self._gpu_device()
            vertices = self.vertices.to(dev)
            faces = self.faces.to(dev)

            mesh = _MeshBackendClass()
            mesh.init(vertices, faces)
            mesh.get_edges()
            mesh.get_boundary_info()
            if mesh.num_boundaries == 0:
                return
            mesh.get_vertex_edge_adjacency()
            mesh.get_vertex_boundary_adjacency()
            mesh.get_manifold_boundary_adjacency()
            mesh.read_manifold_boundary_adjacency()
            mesh.get_boundary_connected_components()
            mesh.get_boundary_loops()
            if mesh.num_boundary_loops == 0:
                return
            mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        elif HAS_TRIMESH:
            tm = _trimesh.Trimesh(
                vertices=self.vertices.cpu().numpy(),
                faces=self.faces.cpu().numpy(),
                process=False,
            )
            _trimesh.repair.fill_holes(tm)
            self.vertices = torch.from_numpy(tm.vertices.astype(np.float32)).to(self.device)
            self.faces = torch.from_numpy(tm.faces.astype(np.int32)).to(self.device)
        else:
            raise RuntimeError("No mesh repair backend available. Install trimesh or cumesh.")

    def remove_faces(self, face_mask: torch.Tensor):
        if _HAS_MESH and self._gpu_device():
            dev = self._gpu_device()
            vertices = self.vertices.to(dev)
            faces = self.faces.to(dev)

            mesh = _MeshBackendClass()
            mesh.init(vertices, faces)
            mesh.remove_faces(face_mask)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        else:
            # CPU fallback: mask out faces and re-index vertices
            keep = ~face_mask.cpu()
            new_faces = self.faces.cpu()[keep]
            used_verts = torch.unique(new_faces)
            vert_map = torch.full((self.vertices.shape[0],), -1, dtype=torch.long)
            vert_map[used_verts] = torch.arange(len(used_verts))
            self.vertices = self.vertices.cpu()[used_verts].to(self.device)
            self.faces = vert_map[new_faces].int().to(self.device)

    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):
        if _HAS_MESH and self._gpu_device():
            dev = self._gpu_device()
            vertices = self.vertices.to(dev)
            faces = self.faces.to(dev)

            mesh = _MeshBackendClass()
            mesh.init(vertices, faces)
            mesh.simplify(target, verbose=verbose, options=options)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        elif HAS_FAST_SIMPLIFICATION:
            verts_np = self.vertices.cpu().numpy().astype(np.float64)
            faces_np = self.faces.cpu().numpy()
            ratio = min(1.0, target / max(faces_np.shape[0], 1))
            new_verts, new_faces = fast_simplification.simplify(
                verts_np, faces_np, target_reduction=(1.0 - ratio)
            )
            self.vertices = torch.from_numpy(new_verts.astype(np.float32)).to(self.device)
            self.faces = torch.from_numpy(new_faces.astype(np.int32)).to(self.device)
        elif HAS_TRIMESH:
            tm = _trimesh.Trimesh(
                vertices=self.vertices.cpu().numpy(),
                faces=self.faces.cpu().numpy(),
                process=False,
            )
            if hasattr(tm, 'simplify_quadric_decimation'):
                tm = tm.simplify_quadric_decimation(target)
            self.vertices = torch.from_numpy(tm.vertices.astype(np.float32)).to(self.device)
            self.faces = torch.from_numpy(tm.faces.astype(np.int32)).to(self.device)
        else:
            if verbose:
                print("Warning: No simplification backend available, skipping.")


class TextureFilterMode:
    CLOSEST = 0
    LINEAR = 1


class TextureWrapMode:
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRRORED_REPEAT = 2


class AlphaMode:
    OPAQUE = 0
    MASK = 1
    BLEND = 2


class Texture:
    def __init__(
        self,
        image: torch.Tensor,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        wrap_mode: TextureWrapMode = TextureWrapMode.REPEAT
    ):
        self.image = image
        self.filter_mode = filter_mode
        self.wrap_mode = wrap_mode

    def to(self, device, non_blocking=False):
        return Texture(
            self.image.to(device, non_blocking=non_blocking),
            self.filter_mode,
            self.wrap_mode,
        )


class PbrMaterial:
    def __init__(
        self,
        base_color_texture: Optional[Texture] = None,
        base_color_factor: Union[torch.Tensor, List[float]] = [1.0, 1.0, 1.0],
        metallic_texture: Optional[Texture] = None,
        metallic_factor: float = 1.0,
        roughness_texture: Optional[Texture] = None,
        roughness_factor: float = 1.0,
        alpha_texture: Optional[Texture] = None,
        alpha_factor: float = 1.0,
        alpha_mode: AlphaMode = AlphaMode.OPAQUE,
        alpha_cutoff: float = 0.5,
    ):
        self.base_color_texture = base_color_texture
        self.base_color_factor = torch.tensor(base_color_factor, dtype=torch.float32)[:3]
        self.metallic_texture = metallic_texture
        self.metallic_factor = metallic_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor
        self.alpha_texture = alpha_texture
        self.alpha_factor = alpha_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff

    def to(self, device, non_blocking=False):
        return PbrMaterial(
            base_color_texture=self.base_color_texture.to(device, non_blocking=non_blocking) if self.base_color_texture is not None else None,
            base_color_factor=self.base_color_factor.to(device, non_blocking=non_blocking),
            metallic_texture=self.metallic_texture.to(device, non_blocking=non_blocking) if self.metallic_texture is not None else None,
            metallic_factor=self.metallic_factor,
            roughness_texture=self.roughness_texture.to(device, non_blocking=non_blocking) if self.roughness_texture is not None else None,
            roughness_factor=self.roughness_factor,
            alpha_texture=self.alpha_texture.to(device, non_blocking=non_blocking) if self.alpha_texture is not None else None,
            alpha_factor=self.alpha_factor,
            alpha_mode=self.alpha_mode,
            alpha_cutoff=self.alpha_cutoff,
        )


class MeshWithPbrMaterial(Mesh):
    def __init__(self,
        vertices,
        faces,
        material_ids,
        uv_coords,
        materials: List[PbrMaterial],
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.material_ids = material_ids    # [M]
        self.uv_coords = uv_coords          # [M, 3, 2]
        self.materials = materials
        self.layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

    def to(self, device, non_blocking=False):
        return MeshWithPbrMaterial(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.material_ids.to(device, non_blocking=non_blocking),
            self.uv_coords.to(device, non_blocking=non_blocking),
            [material.to(device, non_blocking=non_blocking) for material in self.materials],
        )


class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )
        
    def query_attrs(self, xyz):
        if HAS_FLEX_GEMM:
            grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
            vertex_attrs = grid_sample_3d(
                self.attrs,
                torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
                self.voxel_shape,
                grid,
                mode='trilinear'
            )[0]
            return vertex_attrs

        # CPU/MPS fallback: build dense volume then F.grid_sample
        C = self.attrs.shape[-1]
        spatial = self.voxel_shape[2:]  # (D, H, W)
        D, H, W = spatial
        dense_vol = torch.zeros(1, C, D, H, W, dtype=self.attrs.dtype, device=self.attrs.device)
        cx, cy, cz = self.coords[:, 0].long(), self.coords[:, 1].long(), self.coords[:, 2].long()
        dense_vol[0, :, cx, cy, cz] = self.attrs.T

        # Normalize query points to [-1, 1] for F.grid_sample
        grid_pts = ((xyz - self.origin) / self.voxel_size)
        # Normalize to [-1, 1]: grid_sample expects (z, y, x) ordering in the last dim
        grid_pts_norm = torch.stack([
            grid_pts[:, 2] / (W - 1) * 2 - 1,
            grid_pts[:, 1] / (H - 1) * 2 - 1,
            grid_pts[:, 0] / (D - 1) * 2 - 1,
        ], dim=-1)
        grid_pts_norm = grid_pts_norm.reshape(1, 1, 1, -1, 3)
        sampled = F_grid.grid_sample(dense_vol, grid_pts_norm, mode='bilinear', align_corners=True, padding_mode='border')
        # sampled: [1, C, 1, 1, N] -> [N, C]
        vertex_attrs = sampled.reshape(C, -1).T
        return vertex_attrs

    def query_vertex_attrs(self):
        return self.query_attrs(self.vertices)
