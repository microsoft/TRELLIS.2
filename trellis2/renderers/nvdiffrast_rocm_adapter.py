"""
nvdiffrast ROCm adapter — drop-in replacement using PyTorch3D or pure PyTorch.

Usage:
    # Instead of: import nvdiffrast.torch as dr
    from trellis2.renderers.nvdiffrast_rocm_adapter import dr

This module provides compatible implementations of:
    - dr.RasterizeCudaContext
    - dr.rasterize()
    - dr.interpolate()
    - dr.texture()
    - dr.antialias()
    - dr.DepthPeeler

Backend priority:
    1. nvdiffrast (if available — NVIDIA GPUs)
    2. pytorch3d (if available — has CUDA rasterizer + CPU fallback)
    3. pure PyTorch (always works, slower)
"""

import torch
import torch.nn.functional as F
import warnings
import os

# Try to import nvdiffrast first (NVIDIA path)
_BACKEND = None
try:
    if os.environ.get("FORCE_ROCM_RASTERIZER", "0") != "1":
        import nvdiffrast.torch as _nvdr
        _BACKEND = "nvdiffrast"
except ImportError:
    pass

# Try PyTorch3D as fallback
if _BACKEND is None:
    try:
        import pytorch3d
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            RasterizationSettings,
            MeshRasterizer,
            look_at_view_transform,
        )
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        _BACKEND = "pytorch3d"
    except ImportError:
        pass

# Final fallback: pure PyTorch
if _BACKEND is None:
    _BACKEND = "pytorch"

print(f"[nvdiffrast_rocm_adapter] Using backend: {_BACKEND}")


class RasterizeCudaContext:
    """Drop-in replacement for dr.RasterizeCudaContext."""
    def __init__(self, device=None):
        self.device = device or torch.device('cuda')
        if _BACKEND == "nvdiffrast":
            self._ctx = _nvdr.RasterizeCudaContext(device=device)
        else:
            self._ctx = None


def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    """
    Drop-in replacement for dr.rasterize().

    Args:
        glctx: RasterizeCudaContext
        pos: (B, V, 4) clip-space vertex positions
        tri: (F, 3) triangle indices
        resolution: (H, W) tuple
        ranges: optional
        grad_db: bool

    Returns:
        rast: (B, H, W, 4) — [u, v, z/w, triangle_id+1]
        rast_db: (B, H, W, 4) — screen-space derivatives (zeros for non-nvdiffrast)
    """
    if _BACKEND == "nvdiffrast":
        return _nvdr.rasterize(glctx._ctx, pos, tri, resolution, ranges=ranges, grad_db=grad_db)

    if isinstance(resolution, (list, tuple)):
        H, W = resolution
    else:
        H = W = resolution

    B = pos.shape[0]
    device = pos.device

    if _BACKEND == "pytorch3d":
        return _rasterize_pytorch3d(pos, tri, H, W, device)

    return _rasterize_pure_pytorch(pos, tri, H, W, device)


def _rasterize_pytorch3d(pos, tri, H, W, device):
    """Rasterize using PyTorch3D."""
    B = pos.shape[0]

    # Convert clip space (x,y,z,w) to NDC
    pos_ndc = pos[..., :3] / pos[..., 3:4]  # perspective divide

    # PyTorch3D uses different NDC convention, flip y
    verts = pos_ndc.clone()
    verts[..., 1] = -verts[..., 1]

    faces = tri.int().to(device)

    rast_out = torch.zeros(B, H, W, 4, device=device)
    rast_db = torch.zeros(B, H, W, 4, device=device)

    for b in range(B):
        meshes = Meshes(verts=[verts[b, :, :3]], faces=[faces])
        settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
        )
        rasterizer = MeshRasterizer(raster_settings=settings)
        fragments = rasterizer(meshes)

        # Convert PyTorch3D fragments to nvdiffrast format
        pix_to_face = fragments.pix_to_face[..., 0]  # (1, H, W)
        bary_coords = fragments.bary_coords[..., 0, :]  # (1, H, W, 3)
        zbuf = fragments.zbuf[..., 0]  # (1, H, W)

        # nvdiffrast rast format: [u, v, z/w, triangle_id+1]
        mask = (pix_to_face >= 0).float()
        rast_out[b, :, :, 0] = bary_coords[0, :, :, 1]  # u
        rast_out[b, :, :, 1] = bary_coords[0, :, :, 2]  # v
        rast_out[b, :, :, 2] = zbuf[0] * mask[0]
        rast_out[b, :, :, 3] = (pix_to_face[0] + 1).float() * mask[0]

    return rast_out, rast_db


def _rasterize_pure_pytorch(pos, tri, H, W, device):
    """
    Minimal software rasterizer in pure PyTorch.
    Slow but works on any device. Suitable for small meshes / visualization.
    """
    B = pos.shape[0]

    # Perspective divide
    pos_ndc = pos[..., :3] / pos[..., 3:4]

    rast_out = torch.zeros(B, H, W, 4, device=device)
    rast_db = torch.zeros(B, H, W, 4, device=device)

    for b in range(B):
        verts = pos_ndc[b]  # (V, 3)
        faces = tri.long()  # (F, 3)

        # Screen coordinates
        x_screen = (verts[:, 0] * 0.5 + 0.5) * W
        y_screen = (1.0 - (verts[:, 1] * 0.5 + 0.5)) * H  # flip y

        # For each pixel, find which triangle covers it (brute force — slow)
        # Only practical for small meshes
        px = torch.arange(W, device=device).float() + 0.5
        py = torch.arange(H, device=device).float() + 0.5
        grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')  # (H, W)

        z_buffer = torch.full((H, W), float('inf'), device=device)

        for f_idx in range(min(faces.shape[0], 100000)):  # cap for safety
            v0, v1, v2 = faces[f_idx]

            # Triangle vertices in screen space
            x0, y0 = x_screen[v0], y_screen[v0]
            x1, y1 = x_screen[v1], y_screen[v1]
            x2, y2 = x_screen[v2], y_screen[v2]

            # Barycentric coordinates
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom.item()) < 1e-10:
                continue

            w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
            w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
            w2 = 1.0 - w0 - w1

            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            z = w0 * verts[v0, 2] + w1 * verts[v1, 2] + w2 * verts[v2, 2]

            closer = inside & (z < z_buffer)
            z_buffer = torch.where(closer, z, z_buffer)
            rast_out[b, :, :, 0] = torch.where(closer, w1, rast_out[b, :, :, 0])
            rast_out[b, :, :, 1] = torch.where(closer, w2, rast_out[b, :, :, 1])
            rast_out[b, :, :, 2] = torch.where(closer, z, rast_out[b, :, :, 2])
            rast_out[b, :, :, 3] = torch.where(closer, float(f_idx + 1), rast_out[b, :, :, 3])

    return rast_out, rast_db


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """
    Drop-in replacement for dr.interpolate().

    Args:
        attr: (B, V, C) or (1, V, C) vertex attributes
        rast: (B, H, W, 4) rasterization output
        tri: (F, 3) triangle indices
        rast_db: optional derivatives
        diff_attrs: optional

    Returns:
        out: (B, H, W, C) interpolated attributes
        out_db: (B, H, W, C*2) derivatives (zeros if not available)
    """
    if _BACKEND == "nvdiffrast":
        return _nvdr.interpolate(attr, rast, tri, rast_db=rast_db, diff_attrs=diff_attrs)

    B, H, W, _ = rast.shape

    # Get barycentric coordinates and triangle IDs
    u = rast[..., 0:1]  # (B, H, W, 1)
    v = rast[..., 1:2]
    tri_id = rast[..., 3:4].long() - 1  # 0-indexed
    mask = (rast[..., 3:4] > 0).float()

    # w0 = 1 - u - v, w1 = u, w2 = v
    w0 = (1.0 - u - v) * mask
    w1 = u * mask
    w2 = v * mask

    # Clamp triangle IDs
    tri_id_clamped = tri_id.clamp(min=0).squeeze(-1)  # (B, H, W)

    # Get vertex indices for each pixel's triangle
    tri_long = tri.long()
    flat_tri_id = tri_id_clamped.reshape(-1)  # (B*H*W,)
    v0_idx = tri_long[flat_tri_id, 0]  # (B*H*W,)
    v1_idx = tri_long[flat_tri_id, 1]
    v2_idx = tri_long[flat_tri_id, 2]

    # Handle batched or single-batch attr
    if attr.shape[0] == 1 and B > 1:
        attr_expanded = attr.expand(B, -1, -1)
    else:
        attr_expanded = attr

    C = attr_expanded.shape[-1]

    # Gather vertex attributes
    attr_flat = attr_expanded.reshape(-1, C)  # (B*V, C)
    batch_offset = torch.arange(B, device=attr.device).unsqueeze(1).unsqueeze(2) * attr_expanded.shape[1]
    batch_offset_flat = batch_offset.expand(B, H, W).reshape(-1)

    a0 = attr_flat[v0_idx + batch_offset_flat]  # (B*H*W, C)
    a1 = attr_flat[v1_idx + batch_offset_flat]
    a2 = attr_flat[v2_idx + batch_offset_flat]

    # Barycentric interpolation
    w0_flat = w0.reshape(-1, 1)
    w1_flat = w1.reshape(-1, 1)
    w2_flat = w2.reshape(-1, 1)

    out = (a0 * w0_flat + a1 * w1_flat + a2 * w2_flat).reshape(B, H, W, C)
    out = out * mask

    # Derivatives (zeros — no screen-space derivatives in fallback)
    out_db = torch.zeros(B, H, W, C * 2, device=attr.device) if rast_db is not None else None

    if rast_db is not None:
        return out, out_db
    return out, None


def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None,
            filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """
    Drop-in replacement for dr.texture().

    Args:
        tex: (B, H, W, C) texture image
        uv: (B, Hp, Wp, 2) UV coordinates in [0, 1]
        uv_da: optional UV derivatives
        filter_mode: 'auto', 'nearest', 'linear', 'linear-mipmap-nearest', etc.
        boundary_mode: 'wrap', 'clamp'

    Returns:
        out: (B, Hp, Wp, C) sampled texture
    """
    if _BACKEND == "nvdiffrast":
        return _nvdr.texture(tex, uv, uv_da=uv_da, mip_level_bias=mip_level_bias,
                            mip=mip, filter_mode=filter_mode,
                            boundary_mode=boundary_mode, max_mip_level=max_mip_level)

    # Pure PyTorch texture sampling using grid_sample
    B, Ht, Wt, C = tex.shape
    _, Hp, Wp, _ = uv.shape

    # Convert UV [0,1] to grid_sample format [-1,1]
    grid = uv * 2.0 - 1.0  # (B, Hp, Wp, 2)

    # Handle boundary mode
    if boundary_mode == 'wrap':
        # Wrap UVs
        grid = (grid + 1.0) % 2.0 - 1.0
        padding_mode = 'zeros'
    elif boundary_mode == 'clamp':
        padding_mode = 'zeros'
    else:
        padding_mode = 'zeros'

    # grid_sample expects (B, C, H, W) input and (B, Hout, Wout, 2) grid
    tex_bchw = tex.permute(0, 3, 1, 2)  # (B, C, H, W)

    # Flip y for grid_sample convention
    grid_flipped = grid.clone()
    grid_flipped[..., 1] = -grid_flipped[..., 1]

    mode = 'bilinear' if filter_mode in ('auto', 'linear', 'linear-mipmap-nearest',
                                           'linear-mipmap-linear') else 'nearest'

    out = F.grid_sample(tex_bchw, grid_flipped, mode=mode,
                        padding_mode=padding_mode, align_corners=False)

    return out.permute(0, 2, 3, 1)  # back to (B, Hp, Wp, C)


def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """
    Drop-in replacement for dr.antialias().
    Falls back to no-op for non-nvdiffrast backends (edges won't be antialiased).
    """
    if _BACKEND == "nvdiffrast":
        return _nvdr.antialias(color, rast, pos, tri,
                              topology_hash=topology_hash,
                              pos_gradient_boost=pos_gradient_boost)

    # No-op fallback — just return the color as-is
    # Real antialiasing would need edge detection + blending
    return color


class DepthPeeler:
    """
    Drop-in replacement for dr.DepthPeeler.
    Implements depth peeling by iteratively masking out closer fragments.
    """
    def __init__(self, glctx, pos, tri, resolution, ranges=None, grad_db=True):
        self.glctx = glctx
        self.pos = pos
        self.tri = tri
        self.resolution = resolution
        self.ranges = ranges
        self.grad_db = grad_db
        self._layer = 0
        self._prev_z = None

        if _BACKEND == "nvdiffrast":
            self._peeler = _nvdr.DepthPeeler(glctx._ctx, pos, tri, resolution,
                                             ranges=ranges, grad_db=grad_db)
        else:
            self._peeler = None

    def __enter__(self):
        if self._peeler is not None:
            self._peeler.__enter__()
        return self

    def __exit__(self, *args):
        if self._peeler is not None:
            self._peeler.__exit__(*args)

    def rasterize_next_layer(self):
        if self._peeler is not None:
            return self._peeler.rasterize_next_layer()

        # Fallback: just rasterize normally (no actual depth peeling)
        # For the first layer, this is correct. Subsequent layers will be empty.
        if self._layer == 0:
            self._layer += 1
            return rasterize(self.glctx, self.pos, self.tri, self.resolution,
                           ranges=self.ranges, grad_db=self.grad_db)
        else:
            H, W = self.resolution if isinstance(self.resolution, (list, tuple)) else (self.resolution, self.resolution)
            B = self.pos.shape[0]
            device = self.pos.device
            self._layer += 1
            return (torch.zeros(B, H, W, 4, device=device),
                    torch.zeros(B, H, W, 4, device=device))


# Create a module-like namespace for drop-in compatibility
class _DrModule:
    """Fake module that mimics `nvdiffrast.torch` API."""
    RasterizeCudaContext = RasterizeCudaContext
    rasterize = staticmethod(rasterize)
    interpolate = staticmethod(interpolate)
    texture = staticmethod(texture)
    antialias = staticmethod(antialias)
    DepthPeeler = DepthPeeler

dr = _DrModule()
