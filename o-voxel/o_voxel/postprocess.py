from typing import *
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual

import platform
import os

_HAS_DR = False
_HAS_MESH = False
_BACKEND = None
dr = None
_METAL_DISABLED = os.environ.get('TRELLIS_DISABLE_METAL', '0') == '1'
_BACKEND_ERRORS = {}

# Differentiable rasterizer — mtldiffrast (Metal) or nvdiffrast (CUDA)
try:
    if _METAL_DISABLED:
        raise ImportError('Metal disabled by TRELLIS_DISABLE_METAL=1')
    import mtldiffrast.torch as dr
    _HAS_DR = True
    _BACKEND = 'metal'
except (ImportError, RuntimeError, OSError) as exc:
    _BACKEND_ERRORS['mtldiffrast'] = str(exc)
    try:
        import nvdiffrast.torch as dr
        _HAS_DR = True
        _BACKEND = 'cuda'
    except (ImportError, RuntimeError, OSError) as cuda_exc:
        _BACKEND_ERRORS['nvdiffrast'] = str(cuda_exc)

# Mesh processing — cumesh auto-selects Metal/CUDA
try:
    if _METAL_DISABLED and platform.system() == 'Darwin':
        raise ImportError('Metal disabled by TRELLIS_DISABLE_METAL=1')
    import cumesh
    _MeshBackend = cumesh.CuMesh
    _BVH = cumesh.cuBVH
    _remesh_narrow_band_dc = cumesh.remeshing.remesh_narrow_band_dc
    _HAS_MESH = True
    if _BACKEND is None:
        _BACKEND = 'metal' if platform.system() == 'Darwin' else 'cuda'
except (ImportError, RuntimeError, OSError) as exc:
    _BACKEND_ERRORS['cumesh'] = str(exc)

_HAS_GPU_DEPS = _HAS_DR and _HAS_MESH

try:
    if _METAL_DISABLED and platform.system() == 'Darwin':
        raise ImportError('Metal disabled by TRELLIS_DISABLE_METAL=1')
    from flex_gemm.ops.grid_sample import grid_sample_3d as _flex_grid_sample_3d
    _HAS_FLEX_GEMM = True
except (ImportError, RuntimeError, OSError) as exc:
    _BACKEND_ERRORS['flex_gemm'] = str(exc)
    _HAS_FLEX_GEMM = False


def _grid_sample_3d(feats, coords, shape, grid, mode='trilinear'):
    """Grid sampling with flex_gemm on CUDA, F.grid_sample fallback otherwise."""
    if _HAS_FLEX_GEMM:
        return _flex_grid_sample_3d(feats, coords, shape, grid, mode=mode)
    import torch.nn.functional as F_gs
    B, C = shape[0], shape[1]
    D, H, W = shape[2], shape[3], shape[4]
    device = feats.device
    dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=device)
    batch_idx = coords[:, 0].long()
    cx, cy, cz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
    dense_vol[batch_idx, :, cx, cy, cz] = feats
    grid_norm = torch.stack([
        grid[..., 2] / (W - 1) * 2 - 1,
        grid[..., 1] / (H - 1) * 2 - 1,
        grid[..., 0] / (D - 1) * 2 - 1,
    ], dim=-1).reshape(B, 1, 1, -1, 3)
    sampled = F_gs.grid_sample(dense_vol, grid_norm, mode='bilinear',
                               align_corners=True, padding_mode='border')
    M = grid.shape[1]
    return sampled.reshape(B * C, M)


def _guarded_project_back(
    dc_vertices,
    dc_faces,
    src_vertices,
    src_faces,
    bvh,
    strength,
    voxel_size,
    max_dist_voxels=1.5,
    min_normal_agreement=0.5,
    max_iters=3,
    verbose=False,
):
    """Project DC vertices to the source mesh without introducing face flips."""

    dc_face_indices = dc_faces.long()

    def face_crosses(vertex_positions):
        triangles = vertex_positions[dc_face_indices]
        return torch.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1,
        )

    def normalized(vectors):
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        return vectors / lengths.clamp_min(torch.finfo(vectors.dtype).eps)

    dc_face_crosses_before = face_crosses(dc_vertices)
    dc_vertex_normals = torch.zeros_like(dc_vertices)
    for corner in range(3):
        dc_vertex_normals.index_add_(
            0,
            dc_face_indices[:, corner],
            dc_face_crosses_before,
        )
    dc_vertex_normals = normalized(dc_vertex_normals)

    dist, face_id, uvw = bvh.unsigned_distance(dc_vertices, return_uvw=True)
    src_triangles = src_vertices[src_faces[face_id.long()].long()]
    closest = (src_triangles * uvw.unsqueeze(-1)).sum(dim=1)
    src_face_normals = normalized(
        torch.cross(
            src_triangles[:, 1] - src_triangles[:, 0],
            src_triangles[:, 2] - src_triangles[:, 0],
            dim=1,
        )
    )

    normal_agreement = (dc_vertex_normals * src_face_normals).sum(dim=1).abs()
    max_distance = torch.as_tensor(
        max_dist_voxels, dtype=dist.dtype, device=dist.device
    ) * torch.as_tensor(voxel_size, dtype=dist.dtype, device=dist.device)
    min_agreement = torch.as_tensor(
        min_normal_agreement,
        dtype=normal_agreement.dtype,
        device=normal_agreement.device,
    )
    moved = (dist.reshape(-1) <= max_distance) & (
        normal_agreement >= min_agreement
    )
    projection_strength = torch.as_tensor(
        strength, dtype=dc_vertices.dtype, device=dc_vertices.device
    )
    projected_vertices = dc_vertices + projection_strength * (
        closest - dc_vertices
    )
    new_vertices = torch.where(moved.unsqueeze(1), projected_vertices, dc_vertices)
    reverted = torch.zeros_like(moved)
    dc_face_normals_before = normalized(dc_face_crosses_before)

    def bad_faces(vertex_positions):
        crosses_after = face_crosses(vertex_positions)
        normals_after = normalized(crosses_after)
        flipped = (dc_face_normals_before * normals_after).sum(dim=1) < 0
        areas_after = torch.linalg.vector_norm(crosses_after, dim=1) * 0.5
        return flipped | (areas_after < 1e-14)

    def revert_bad_face_vertices(vertex_positions, bad):
        nonlocal moved, reverted
        affected = torch.zeros_like(moved)
        affected[dc_face_indices[bad].reshape(-1)] = True
        reverted |= moved & affected
        moved &= ~affected
        return torch.where(affected.unsqueeze(1), dc_vertices, vertex_positions)

    for _ in range(max(0, int(max_iters))):
        bad = bad_faces(new_vertices)
        if not bool(bad.any()):
            break
        new_vertices = revert_bad_face_vertices(new_vertices, bad)

    bad = bad_faces(new_vertices)
    if bool(bad.any()):
        new_vertices = revert_bad_face_vertices(new_vertices, bad)

    moved_count = int(moved.sum().item())
    reverted_count = int(reverted.sum().item())
    if verbose:
        print(
            f"Guarded projection: {moved_count} vertices moved, "
            f"{reverted_count} reverted"
        )
    return new_vertices, moved_count, reverted_count


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
    remesh: bool = True,
    remesh_band: float = 1,
    # Projection recovers sub-voxel detail lost by DC remeshing. Distance,
    # normal-agreement, and face-flip guards keep dirty source sheets from
    # pulling the rebuilt surface into speckles.
    remesh_project: float = 0.7,
    remesh_project_max_dist: float = 1.5,
    remesh_project_min_agreement: float = 0.5,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sprase tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target face count, or None to preserve full resolution
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        remesh_project_max_dist: maximum projection distance in remesh voxels
        remesh_project_min_agreement: minimum absolute source/DC normal
            agreement for projection
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    # Auto-fallback to CPU pipeline when no GPU deps available
    if not _HAS_GPU_DEPS:
        from .postprocess_cpu import to_glb as to_glb_cpu
        return to_glb_cpu(
            vertices=vertices, faces=faces, attr_volume=attr_volume,
            coords=coords, attr_layout=attr_layout, aabb=aabb,
            voxel_size=voxel_size, grid_size=grid_size,
            decimation_target=decimation_target, texture_size=texture_size,
            remesh=remesh, remesh_band=remesh_band, remesh_project=remesh_project,
            remesh_project_max_dist=remesh_project_max_dist,
            remesh_project_min_agreement=remesh_project_min_agreement,
            verbose=verbose, use_tqdm=use_tqdm,
        )

    # Select device based on backend
    # Metal path: all GPU compute goes through Metal kernels directly (mtldiffrast,
    # mtlbvh, cumesh, flex_gemm). CPU tensors on Apple Silicon unified memory are
    # directly GPU-accessible — no MPS overhead needed.
    if _BACKEND == 'metal':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    effective_decimation_target = (
        int(faces.shape[0]) if decimation_target is None else int(decimation_target)
    )
    if effective_decimation_target <= 0:
        effective_decimation_target = int(faces.shape[0])

    # Move data to GPU
    vertices = vertices.to(device)
    faces = faces.to(device)

    # Initialize mesh handler
    mesh = _MeshBackend()
    mesh.init(vertices, faces)
    
    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    if use_tqdm:
        pbar.update(1)
        
    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = _BVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")
    
    # --- Branch 1: Legacy Pipeline (Simplification & Cleaning) ---
    if not remesh:
        # Step 1: Aggressive simplification (3x target)
        mesh.simplify(effective_decimation_target * 3, verbose=verbose)
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 3: Final simplification to target count
        mesh.simplify(effective_decimation_target, verbose=verbose)
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 5: Unify face orientations
        mesh.unify_face_orientations()
    
    # --- Branch 2: Production Remeshing Pipeline ---
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        remesh_domain_scale = (resolution + 3 * remesh_band) / resolution * scale
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        remeshed_vertices, remeshed_faces = _remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = remesh_domain_scale,
            resolution = resolution,
            band = remesh_band,
            project_back = 0,
            verbose = verbose,
            bvh = bvh,
        )
        if remesh_project > 0:
            remeshed_vertices, _, _ = _guarded_project_back(
                remeshed_vertices,
                remeshed_faces,
                vertices,
                faces,
                bvh,
                strength=remesh_project,
                voxel_size=remesh_domain_scale / resolution,
                max_dist_voxels=remesh_project_max_dist,
                min_normal_agreement=remesh_project_min_agreement,
                verbose=verbose,
            )
        mesh.init(remeshed_vertices, remeshed_faces)
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # repair_non_manifold_edges() is intentionally excluded here: on the
        # measured ak74m body it regressed boundary edges 119 -> 500 and broke
        # winding by vertex-splitting junctions into open seams.
        mesh.remove_duplicate_faces()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
        if verbose:
            print(f"After cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        if decimation_target is not None:
            mesh.simplify(effective_decimation_target, verbose=verbose)
            mesh.fill_holes(max_hole_perimeter=3e-2)
            mesh.unify_face_orientations()
            if verbose:
                print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    
    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")
    
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.to(device)
    out_faces = out_faces.to(device)
    out_uvs = out_uvs.to(device)
    out_vmaps = out_vmaps.to(device)
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)
        
    # Setup differentiable rasterizer context
    ctx = dr.MtlRasterizeContext() if _BACKEND == 'metal' else dr.RasterizeCudaContext()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device=device, dtype=torch.float32)
    
    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)
    
    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0
    
    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    
    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    
    # Trilinear sampling from the attribute volume (Color, Material props)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
    attrs[mask] = _grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    mask = mask.cpu().numpy()
    
    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    # Auto-detect transparency from baked alpha values
    alpha_valid = alpha[mask]
    if alpha_valid.size > 0 and alpha_valid.min() < 250:
        alpha_mode = 'BLEND'
        if verbose:
            print(f"Detected transparency (alpha min={alpha_valid.min()}), using BLEND mode")
    else:
        alpha_mode = 'OPAQUE'
    
    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    
    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )
    
    # --- Coordinate System Conversion & Final Object ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    
    # Y-up to Z-up for GLB (must copy to avoid in-place corruption)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
    uvs_np[:, 1] = 1 - uvs_np[:, 1]
    
    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")
    
    return textured_mesh
