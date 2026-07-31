"""
macOS GLB export pipeline.

Replaces cumesh + nvdiffrast + flex_gemm with:
- fast_simplification / trimesh for mesh decimation
- xatlas for UV unwrapping
- PyTorch MPS-accelerated UV rasterization + texture baking
- OpenCV cv2.inpaint for texture inpainting
"""
from typing import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import warnings
from PIL import Image
import trimesh
import trimesh.visual
import xatlas

try:
    import fast_simplification
    HAS_FAST_SIMPLIFICATION = True
except ImportError:
    HAS_FAST_SIMPLIFICATION = False


def _get_device():
    """Get best available device: MPS > CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _rasterize_uv_gpu(vertices, faces, uvs, texture_size, device=None):
    """
    GPU-accelerated UV-space rasterization via vectorized PyTorch.

    For each face, computes barycentric coords for all texels in its bounding box
    in parallel on MPS/CPU. Replaces the slow Python double-loop.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices (int64)
        uvs: (V, 2) per-vertex UV coords in [0, 1]
        texture_size: int
        device: torch device (defaults to MPS if available)

    Returns:
        pos: (H, W, 3) interpolated 3D positions
        mask: (H, W) bool - valid texels
    """
    if device is None:
        device = _get_device()

    H = W = texture_size

    # Move to device
    verts = vertices.float().to(device)
    faces_t = faces.long().to(device)
    uv = uvs.float().to(device)

    # Gather per-face data: (F, 3, 2) UVs and (F, 3, 3) positions
    face_uvs = uv[faces_t]      # (F, 3, 2)
    face_verts = verts[faces_t]  # (F, 3, 3)

    # Scale UVs to pixel coords
    face_uvs_px = face_uvs.clone()
    face_uvs_px[..., 0] *= (W - 1)
    face_uvs_px[..., 1] *= (H - 1)

    # Output buffers
    pos_buf = torch.zeros(H, W, 3, device=device)
    mask_buf = torch.zeros(H, W, dtype=torch.bool, device=device)
    # Use a depth buffer (face index) to handle overlaps — last writer wins like nvdiffrast

    # Process in chunks to avoid OOM
    num_faces = faces_t.shape[0]
    chunk_size = 50000

    for start in range(0, num_faces, chunk_size):
        end = min(start + chunk_size, num_faces)
        chunk_uvs = face_uvs_px[start:end]   # (C, 3, 2)
        chunk_verts = face_verts[start:end]   # (C, 3, 3)
        C = chunk_uvs.shape[0]

        # Bounding boxes per face: (C,)
        bb_min_x = chunk_uvs[..., 0].min(dim=1).values.floor().clamp(min=0).int()
        bb_max_x = chunk_uvs[..., 0].max(dim=1).values.ceil().clamp(max=W - 1).int()
        bb_min_y = chunk_uvs[..., 1].min(dim=1).values.floor().clamp(min=0).int()
        bb_max_y = chunk_uvs[..., 1].max(dim=1).values.ceil().clamp(max=H - 1).int()

        # Compute max bbox size in this chunk to allocate grid
        max_w = (bb_max_x - bb_min_x + 1).max().item()
        max_h = (bb_max_y - bb_min_y + 1).max().item()

        if max_w <= 0 or max_h <= 0:
            continue

        # Create local pixel grids for each face: (C, max_h, max_w, 2)
        # Use offsets from bb_min
        local_y = torch.arange(max_h, device=device).float()
        local_x = torch.arange(max_w, device=device).float()
        gy, gx = torch.meshgrid(local_y, local_x, indexing='ij')  # (max_h, max_w)
        grid = torch.stack([gx, gy], dim=-1)  # (max_h, max_w, 2)
        grid = grid.unsqueeze(0).expand(C, -1, -1, -1).clone()  # (C, max_h, max_w, 2)

        # Offset to absolute pixel coords
        grid[..., 0] += bb_min_x.float().view(C, 1, 1)
        grid[..., 1] += bb_min_y.float().view(C, 1, 1)

        # Validity mask: within bounding box
        valid = (
            (grid[..., 0] <= bb_max_x.float().view(C, 1, 1)) &
            (grid[..., 1] <= bb_max_y.float().view(C, 1, 1)) &
            (grid[..., 0] >= 0) & (grid[..., 0] < W) &
            (grid[..., 1] >= 0) & (grid[..., 1] < H)
        )  # (C, max_h, max_w)

        # Compute barycentric coordinates for each texel relative to each face
        # v0 = uv1 - uv0, v1 = uv2 - uv0, v2 = p - uv0
        uv0 = chunk_uvs[:, 0, :]  # (C, 2)
        v0 = chunk_uvs[:, 1, :] - uv0  # (C, 2)
        v1 = chunk_uvs[:, 2, :] - uv0  # (C, 2)

        # (C, max_h, max_w, 2)
        v2 = grid - uv0.view(C, 1, 1, 2)

        # Barycentric via dot products — all vectorized
        dot00 = (v0 * v0).sum(dim=1)  # (C,)
        dot01 = (v0 * v1).sum(dim=1)  # (C,)
        dot11 = (v1 * v1).sum(dim=1)  # (C,)

        dot02 = (v2 * v0.view(C, 1, 1, 2)).sum(dim=-1)  # (C, max_h, max_w)
        dot12 = (v2 * v1.view(C, 1, 1, 2)).sum(dim=-1)  # (C, max_h, max_w)

        inv_denom = dot00 * dot11 - dot01 * dot01  # (C,)
        # Skip degenerate triangles
        non_degen = inv_denom.abs() > 1e-10
        inv_denom = torch.where(non_degen, 1.0 / inv_denom.clamp(min=1e-10), torch.zeros_like(inv_denom))

        u = (dot11.view(C, 1, 1) * dot02 - dot01.view(C, 1, 1) * dot12) * inv_denom.view(C, 1, 1)
        v = (dot00.view(C, 1, 1) * dot12 - dot01.view(C, 1, 1) * dot02) * inv_denom.view(C, 1, 1)
        w = 1.0 - u - v

        # Inside triangle test (with small epsilon for edge cases)
        eps = -1e-4
        inside = (u >= eps) & (v >= eps) & ((u + v) <= 1.0 - eps) & valid & non_degen.view(C, 1, 1)

        # Interpolate 3D positions: w*v0 + u*v1 + v*v2
        # chunk_verts: (C, 3, 3) — positions of 3 verts per face
        interp_pos = (
            w.unsqueeze(-1) * chunk_verts[:, 0:1, :].unsqueeze(2) +  # (C, 1, 1, 3)
            u.unsqueeze(-1) * chunk_verts[:, 1:2, :].unsqueeze(2) +
            v.unsqueeze(-1) * chunk_verts[:, 2:3, :].unsqueeze(2)
        )  # (C, max_h, max_w, 3)

        # Write to output buffer
        # Get absolute pixel coords for valid texels
        abs_x = grid[..., 0].long()
        abs_y = grid[..., 1].long()

        # Flatten and filter
        inside_flat = inside.reshape(-1)
        abs_x_flat = abs_x.reshape(-1)[inside_flat]
        abs_y_flat = abs_y.reshape(-1)[inside_flat]
        pos_flat = interp_pos.reshape(-1, 3)[inside_flat]

        if abs_x_flat.numel() > 0:
            pos_buf[abs_y_flat, abs_x_flat] = pos_flat
            mask_buf[abs_y_flat, abs_x_flat] = True

    return pos_buf.cpu().numpy(), mask_buf.cpu().numpy()


def _grid_sample_3d_gpu(attr_volume, coords, grid_size, query_pts, voxel_size, aabb, device=None):
    """
    GPU-accelerated trilinear sampling from sparse attribute volume.
    Uses F.grid_sample on MPS.
    """
    if device is None:
        device = _get_device()

    C = attr_volume.shape[1]
    D, H, W = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

    # Build dense volume on device
    attr_vol = attr_volume.float().to(device)
    coords_d = coords.long().to(device)

    dense_vol = torch.zeros(1, C, D, H, W, dtype=torch.float32, device=device)
    cx, cy, cz = coords_d[:, 0], coords_d[:, 1], coords_d[:, 2]
    dense_vol[0, :, cx, cy, cz] = attr_vol.T

    # Normalize query pts to [-1, 1] for grid_sample
    query = query_pts.float().to(device)
    aabb_d = aabb.float().to(device)
    vs_d = voxel_size.float().to(device)

    grid_pts = (query - aabb_d[0]) / vs_d
    grid_pts_norm = torch.stack([
        grid_pts[:, 2] / (W - 1) * 2 - 1,
        grid_pts[:, 1] / (H - 1) * 2 - 1,
        grid_pts[:, 0] / (D - 1) * 2 - 1,
    ], dim=-1)
    grid_pts_norm = grid_pts_norm.reshape(1, 1, 1, -1, 3)

    sampled = F.grid_sample(dense_vol, grid_pts_norm, mode='bilinear', align_corners=True, padding_mode='border')
    return sampled.reshape(C, -1).T.cpu()


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: Optional[int] = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    verbose: bool = False,
    use_tqdm: bool = False,
    **kwargs,
):
    """
    macOS GLB export. Replaces the CUDA pipeline (cumesh + nvdiffrast + flex_gemm)
    with fast_simplification + xatlas + PyTorch MPS rasterization.
    """
    if remesh:
        warnings.warn(
            "kdtree/CPU path cannot remesh; candidate will not be promotable",
            RuntimeWarning,
            stacklevel=2,
        )

    device = _get_device()
    if verbose:
        print(f"Using device: {device}")

    # --- Input Normalization ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32)
    aabb = aabb.cpu()

    if voxel_size is not None:
        if isinstance(voxel_size, (int, float)):
            voxel_size = [voxel_size] * 3
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32)
        voxel_size = (aabb[1] - aabb[0]) / grid_size.float()

    if use_tqdm:
        pbar = tqdm(total=5, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    vertices = vertices.cpu()
    faces = faces.cpu()

    # --- Step 1: Mesh Cleaning ---
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    tm = trimesh.Trimesh(
        vertices=vertices.numpy(),
        faces=faces.numpy(),
        process=False,
    )
    trimesh.repair.fill_holes(tm)
    trimesh.repair.fix_normals(tm)
    if verbose:
        print(f"After hole filling: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    # --- Step 2: Simplification ---
    effective_decimation_target = len(tm.faces) if decimation_target is None else int(decimation_target)
    if effective_decimation_target <= 0:
        effective_decimation_target = len(tm.faces)
    if effective_decimation_target < len(tm.faces):
        if HAS_FAST_SIMPLIFICATION:
            ratio = min(1.0, effective_decimation_target / max(len(tm.faces), 1))
            new_verts, new_faces = fast_simplification.simplify(
                tm.vertices.astype(np.float64), tm.faces,
                target_reduction=(1.0 - ratio),
            )
            tm = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        elif hasattr(tm, 'simplify_quadric_decimation'):
            tm = tm.simplify_quadric_decimation(effective_decimation_target)
        if verbose:
            print(f"After simplification: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    trimesh.repair.fill_holes(tm)
    trimesh.repair.fix_normals(tm)

    if use_tqdm:
        pbar.update(1)

    # --- Step 3: UV Unwrapping with xatlas ---
    if use_tqdm:
        pbar.set_description("UV unwrapping")
    if verbose:
        print("UV unwrapping with xatlas...")

    vmapping, out_faces_np, out_uvs_np = xatlas.parametrize(
        tm.vertices.astype(np.float32),
        tm.faces.astype(np.uint32),
    )
    out_vertices_np = tm.vertices[vmapping].astype(np.float32)
    out_normals_np = tm.vertex_normals[vmapping].astype(np.float32)

    if verbose:
        print(f"After UV: {out_vertices_np.shape[0]} vertices, {out_faces_np.shape[0]} faces")
    if use_tqdm:
        pbar.update(1)

    # --- Step 4: Texture Baking (GPU-accelerated) ---
    if use_tqdm:
        pbar.set_description("Baking textures (MPS)" if device.type == 'mps' else "Baking textures")
    if verbose:
        print("Baking textures...")

    out_vertices_t = torch.from_numpy(out_vertices_np)
    out_faces_t = torch.from_numpy(out_faces_np.astype(np.int64))
    out_uvs_t = torch.from_numpy(out_uvs_np)

    # GPU rasterization in UV space
    pos, mask = _rasterize_uv_gpu(out_vertices_t, out_faces_t, out_uvs_t, texture_size, device)

    valid_pos = torch.from_numpy(pos[mask]).float()

    # Sample attributes from volume (GPU-accelerated)
    attr_volume = attr_volume.cpu()
    coords = coords.cpu()
    attrs_full = torch.zeros(texture_size, texture_size, attr_volume.shape[1])
    if valid_pos.shape[0] > 0:
        sampled = _grid_sample_3d_gpu(attr_volume, coords, grid_size, valid_pos, voxel_size, aabb, device)
        attrs_full[mask] = sampled

    if use_tqdm:
        pbar.update(1)

    # --- Step 5: Texture Post-Processing ---
    if use_tqdm:
        pbar.set_description("Post-processing textures")
    if verbose:
        print("Post-processing textures...")

    mask_inv = (~mask).astype(np.uint8)

    base_color = np.clip(attrs_full[..., attr_layout['base_color']].numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs_full[..., attr_layout['metallic']].numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs_full[..., attr_layout['roughness']].numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs_full[..., attr_layout['alpha']].numpy() * 255, 0, 255).astype(np.uint8)

    # Auto-detect transparency from baked alpha values
    alpha_valid = alpha[mask]
    if alpha_valid.size > 0 and alpha_valid.min() < 250:
        alpha_mode = 'BLEND'
        if verbose:
            print(f"Detected transparency (alpha min={alpha_valid.min()}), using BLEND mode")
    else:
        alpha_mode = 'OPAQUE'

    # Inpainting to fill UV seams
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    if use_tqdm:
        pbar.update(1)

    # --- Step 6: Build PBR Material & Export ---
    if use_tqdm:
        pbar.set_description("Finalizing")
    if verbose:
        print("Building PBR material...")

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
        ),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )

    # Coordinate system conversion (Y-up to Z-up for GLB)
    vertices_out = out_vertices_np.copy()
    normals_out = out_normals_np.copy()
    vertices_out[:, 1], vertices_out[:, 2] = out_vertices_np[:, 2].copy(), -out_vertices_np[:, 1].copy()
    normals_out[:, 1], normals_out[:, 2] = out_normals_np[:, 2].copy(), -out_normals_np[:, 1].copy()
    uvs_out = out_uvs_np.copy()
    uvs_out[:, 1] = 1 - uvs_out[:, 1]

    textured_mesh = trimesh.Trimesh(
        vertices=vertices_out,
        faces=out_faces_np,
        vertex_normals=normals_out,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_out, material=material),
    )

    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    return textured_mesh
