from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from .. import _C

FloatArrayLike = Union[float, List[Any], Tuple[Any, ...], np.ndarray, torch.Tensor]
IntArrayLike = Union[int, List[Any], Tuple[Any, ...], np.ndarray, torch.Tensor]
ArrayLike = Union[List[Any], Tuple[Any, ...], np.ndarray, torch.Tensor]
_EDGE_NEIGHBOR_VOXEL_OFFSET: Dict[torch.device, torch.Tensor] = {}
_QUAD_SPLIT_1: Dict[torch.device, torch.Tensor] = {}
_QUAD_SPLIT_2: Dict[torch.device, torch.Tensor] = {}
_QUAD_SPLIT_TRAIN: Dict[torch.device, torch.Tensor] = {}

__all__ = [
    "mesh_to_flexible_dual_grid",
    "intersect_occ",
    "flexible_dual_grid_to_mesh",
]


def _init_hashmap(grid_size, capacity, device):
    """Create the sparse voxel lookup table used when converting a dual grid to a mesh."""
    VOL = (grid_size[0] * grid_size[1] * grid_size[2]).item()

    # If the number of elements in the tensor is less than 2^32, use uint32 as the hashmap type, otherwise use uint64.
    if VOL < 2**32:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif VOL < 2**64:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap. Get volumn {VOL} > 2^64.")

    hashmap_vals = torch.empty((capacity,), dtype=torch.uint32, device=device)

    return hashmap_keys, hashmap_vals


@torch.no_grad()
def mesh_to_flexible_dual_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_size: Optional[FloatArrayLike] = None,
    grid_size: Optional[IntArrayLike] = None,
    aabb: Optional[ArrayLike] = None,
    face_weight: float = 1.0,
    boundary_weight: float = 1.0,
    regularization_weight: float = 0.1,
    timing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a triangle mesh into a sparse flexible dual grid.

    The mesh is first placed in a voxel grid, then intersection, face, and
    boundary QEF terms are accumulated for the active voxels. The final dual
    vertex in each voxel is found by solving the accumulated QEF. CUDA inputs use
    the CUDA implementation; CPU inputs use the CPU implementation with the same
    public semantics.

    Args:
        vertices (torch.Tensor): The vertices of the mesh.
        faces (torch.Tensor): The faces of the mesh.
        voxel_size (float, list, tuple, np.ndarray, torch.Tensor): The size of each voxel.
        grid_size (int, list, tuple, np.ndarray, torch.Tensor): The size of the grid.
            NOTE: One of voxel_size and grid_size must be provided.
        aabb (list, tuple, np.ndarray, torch.Tensor): The axis-aligned bounding box of the mesh.
            If not provided, it will be computed automatically.
        face_weight (float): The weight of the face term in the QEF when solving the dual vertices.
        boundary_weight (float): The weight of the boundary term in the QEF when solving the dual vertices.
        regularization_weight (float): The weight of the regularization term in the QEF when solving the dual vertices.
        timing (bool): Whether to time the voxelization process.

    Returns:
        torch.Tensor: The indices of the voxels that are occupied by the mesh.
            The shape of the tensor is (N, 3), where N is the number of occupied voxels.
        torch.Tensor: The dual vertices of the mesh.
        torch.Tensor: The intersected flag of each voxel.
    """

    assert isinstance(vertices, torch.Tensor), f"vertices must be a torch.Tensor, but got {type(vertices)}"
    assert isinstance(faces, torch.Tensor), f"faces must be a torch.Tensor, but got {type(faces)}"
    assert vertices.dim() == 2, f"vertices must be a 2D tensor, but got {vertices.shape}"
    assert vertices.size(1) == 3, f"vertices must have 3 columns, but got {vertices.size(1)}"
    assert faces.dim() == 2, f"faces must be a 2D tensor, but got {faces.shape}"
    assert faces.size(1) == 3, f"faces must have 3 columns, but got {faces.size(1)}"
    assert vertices.device == faces.device, "vertices and faces must be on the same device"
    assert voxel_size is not None or grid_size is not None, "Either voxel_size or grid_size must be provided"

    device = vertices.device
    vertices = vertices.to(device=device, dtype=torch.float32).contiguous()
    faces = faces.to(device=device, dtype=torch.int32).contiguous()
    voxel_size_cpu = None
    grid_size_cpu = None
    aabb_cpu = None

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size_cpu = torch.tensor(voxel_size, dtype=torch.float32)
            voxel_size = voxel_size_cpu
        else:
            assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
            assert not voxel_size.is_cuda, "voxel_size Tensor metadata must be on CPU; pass Python values to avoid implicit CUDA sync"
            voxel_size_cpu = voxel_size.to(dtype=torch.float32).contiguous()
        assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
        voxel_size = voxel_size_cpu.to(device=device).contiguous()
        assert voxel_size.dim() == 1, f"voxel_size must be a 1D tensor, but got {voxel_size.shape}"
        assert voxel_size.size(0) == 3, f"voxel_size must have 3 elements, but got {voxel_size.size(0)}"

    if grid_size is not None:
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size_cpu = torch.tensor(grid_size, dtype=torch.int32)
            grid_size = grid_size_cpu
        else:
            assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
            assert not grid_size.is_cuda, "grid_size Tensor metadata must be on CPU; pass Python values to avoid implicit CUDA sync"
            grid_size_cpu = grid_size.to(dtype=torch.int32).contiguous()
        assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
        grid_size = grid_size_cpu.to(device=device).contiguous()
        assert grid_size.dim() == 1, f"grid_size must be a 1D tensor, but got {grid_size.shape}"
        assert grid_size.size(0) == 3, f"grid_size must have 3 elements, but got {grid_size.size(0)}"

    if aabb is not None:
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb_cpu = torch.tensor(aabb, dtype=torch.float32)
            aabb = aabb_cpu
        else:
            assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
            assert not aabb.is_cuda, "aabb Tensor metadata must be on CPU; pass Python values to avoid implicit CUDA sync"
            aabb_cpu = aabb.to(dtype=torch.float32).contiguous()
        assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
        aabb = aabb_cpu.to(device=device).contiguous()
        assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
        assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
        assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Auto adjust aabb
    if aabb is None:
        min_xyz = vertices.min(dim=0).values
        max_xyz = vertices.max(dim=0).values

        if voxel_size is not None:
            padding = torch.ceil((max_xyz - min_xyz) / voxel_size) * voxel_size - (max_xyz - min_xyz)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5
        if grid_size is not None:
            padding = (max_xyz - min_xyz) / (grid_size - 1)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5

        aabb = torch.stack([min_xyz, max_xyz], dim=0).float().contiguous()

    # Fill voxel size or grid size
    if voxel_size_cpu is None:
        assert grid_size_cpu is not None
        if aabb_cpu is None:
            assert not aabb.is_cuda, "CUDA inputs require CPU aabb or explicit voxel_size to avoid implicit CUDA sync"
            aabb_cpu = aabb.detach()
        voxel_size_cpu = ((aabb_cpu[1] - aabb_cpu[0]) / grid_size_cpu).to(dtype=torch.float32).contiguous()
        voxel_size = voxel_size_cpu.to(device=device).contiguous()
    if grid_size_cpu is None:
        assert voxel_size_cpu is not None
        if aabb_cpu is None:
            assert not aabb.is_cuda, "CUDA inputs require CPU aabb or explicit grid_size to avoid implicit CUDA sync"
            aabb_cpu = aabb.detach()
        grid_size_cpu = ((aabb_cpu[1] - aabb_cpu[0]) / voxel_size_cpu).round().to(dtype=torch.int32).contiguous()
        grid_size = grid_size_cpu.to(device=device).contiguous()
    grid_size = grid_size_cpu.to(device=device).contiguous()
    voxel_size_arg = [float(x) for x in voxel_size_cpu.tolist()]
    grid_range_arg = [0, 0, 0] + [int(x) for x in grid_size_cpu.tolist()]

    # Shift mesh vertices into grid-local coordinates before calling C++/CUDA.
    vertices = vertices - aabb[0].reshape(1, 3)

    if vertices.is_cuda:
        ret = _C.mesh_to_flexible_dual_grid_cuda(
            vertices,
            faces,
            voxel_size_arg,
            grid_range_arg,
            face_weight,
            boundary_weight,
            regularization_weight,
        )
    else:
        ret = _C.mesh_to_flexible_dual_grid_cpu(
            vertices,
            faces,
            voxel_size_arg,
            grid_range_arg,
            face_weight,
            boundary_weight,
            regularization_weight,
            timing,
        )

    return ret


@torch.no_grad()
def intersect_occ(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_size: Optional[FloatArrayLike] = None,
    grid_size: Optional[IntArrayLike] = None,
    aabb: Optional[ArrayLike] = None,
) -> torch.Tensor:
    """
    Return only the voxel coordinates intersected by a triangle mesh.

    This uses the same grid setup as mesh_to_flexible_dual_grid, but stops after
    the occupancy stage and does not compute QEFs or dual vertices. It is the
    matching public API for users who only need occupied voxels.
    """

    assert isinstance(vertices, torch.Tensor), f"vertices must be a torch.Tensor, but got {type(vertices)}"
    assert isinstance(faces, torch.Tensor), f"faces must be a torch.Tensor, but got {type(faces)}"
    assert vertices.dim() == 2, f"vertices must be a 2D tensor, but got {vertices.shape}"
    assert vertices.size(1) == 3, f"vertices must have 3 columns, but got {vertices.size(1)}"
    assert faces.dim() == 2, f"faces must be a 2D tensor, but got {faces.shape}"
    assert faces.size(1) == 3, f"faces must have 3 columns, but got {faces.size(1)}"
    assert vertices.device == faces.device, "vertices and faces must be on the same device"
    assert voxel_size is not None or grid_size is not None, "Either voxel_size or grid_size must be provided"

    device = vertices.device
    vertices = vertices.to(device=device, dtype=torch.float32).contiguous()
    faces = faces.to(device=device, dtype=torch.int32).contiguous()
    voxel_size_cpu = None
    grid_size_cpu = None
    aabb_cpu = None

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size_cpu = torch.tensor(voxel_size, dtype=torch.float32)
            voxel_size = voxel_size_cpu
        else:
            assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
            assert not voxel_size.is_cuda, "voxel_size Tensor metadata must be on CPU; pass Python values to avoid implicit CUDA sync"
            voxel_size_cpu = voxel_size.to(dtype=torch.float32).contiguous()
        assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
        voxel_size = voxel_size_cpu.to(device=device).contiguous()
        assert voxel_size.dim() == 1, f"voxel_size must be a 1D tensor, but got {voxel_size.shape}"
        assert voxel_size.size(0) == 3, f"voxel_size must have 3 elements, but got {voxel_size.size(0)}"

    if grid_size is not None:
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size_cpu = torch.tensor(grid_size, dtype=torch.int32)
            grid_size = grid_size_cpu
        else:
            assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
            assert not grid_size.is_cuda, "grid_size Tensor metadata must be on CPU; pass Python values to avoid implicit CUDA sync"
            grid_size_cpu = grid_size.to(dtype=torch.int32).contiguous()
        assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
        grid_size = grid_size_cpu.to(device=device).contiguous()
        assert grid_size.dim() == 1, f"grid_size must be a 1D tensor, but got {grid_size.shape}"
        assert grid_size.size(0) == 3, f"grid_size must have 3 elements, but got {grid_size.size(0)}"

    if aabb is not None:
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb_cpu = torch.tensor(aabb, dtype=torch.float32)
            aabb = aabb_cpu
        else:
            assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
            assert not aabb.is_cuda, "aabb Tensor metadata must be on CPU; pass Python values to avoid implicit CUDA sync"
            aabb_cpu = aabb.to(dtype=torch.float32).contiguous()
        assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
        aabb = aabb_cpu.to(device=device).contiguous()
        assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
        assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
        assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    if aabb is None:
        min_xyz = vertices.min(dim=0).values
        max_xyz = vertices.max(dim=0).values

        if voxel_size is not None:
            padding = torch.ceil((max_xyz - min_xyz) / voxel_size) * voxel_size - (max_xyz - min_xyz)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5
        if grid_size is not None:
            padding = (max_xyz - min_xyz) / (grid_size - 1)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5

        aabb = torch.stack([min_xyz, max_xyz], dim=0).float().contiguous()

    if voxel_size_cpu is None:
        assert grid_size_cpu is not None
        if aabb_cpu is None:
            assert not aabb.is_cuda, "CUDA inputs require CPU aabb or explicit voxel_size to avoid implicit CUDA sync"
            aabb_cpu = aabb.detach()
        voxel_size_cpu = ((aabb_cpu[1] - aabb_cpu[0]) / grid_size_cpu).to(dtype=torch.float32).contiguous()
        voxel_size = voxel_size_cpu.to(device=device).contiguous()
    if grid_size_cpu is None:
        assert voxel_size_cpu is not None
        if aabb_cpu is None:
            assert not aabb.is_cuda, "CUDA inputs require CPU aabb or explicit grid_size to avoid implicit CUDA sync"
            aabb_cpu = aabb.detach()
        grid_size_cpu = ((aabb_cpu[1] - aabb_cpu[0]) / voxel_size_cpu).round().to(dtype=torch.int32).contiguous()
        grid_size = grid_size_cpu.to(device=device).contiguous()
    grid_size = grid_size_cpu.to(device=device).contiguous()
    voxel_size_arg = [float(x) for x in voxel_size_cpu.tolist()]
    grid_range_arg = [0, 0, 0] + [int(x) for x in grid_size_cpu.tolist()]

    vertices = vertices - aabb[0].reshape(1, 3)
    triangles = vertices[faces.to(dtype=torch.long)].contiguous()

    if vertices.is_cuda:
        return _C.intersect_occ_cuda(triangles, voxel_size_arg, grid_range_arg)
    else:
        return _C.intersect_occ_cpu(triangles, voxel_size_arg, grid_range_arg)


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: Optional[torch.Tensor],
    aabb: ArrayLike,
    voxel_size: Optional[FloatArrayLike] = None,
    grid_size: Optional[IntArrayLike] = None,
    train: bool = False,
):
    """
    Extract a triangle mesh from sparse flexible dual grid outputs.

    The function looks up neighboring active voxels around each intersected grid
    edge, forms one quad from the four neighboring dual vertices, then splits
    each quad into triangles. The sparse voxel lookup is built in PyTorch and the
    returned vertices are moved back from grid-local coordinates into the input
    AABB.

    Args:
        coords (torch.Tensor): The coordinates of the voxels.
        dual_vertices (torch.Tensor): The dual vertices.
        intersected_flag (torch.Tensor): The intersected flag.
        split_weight (torch.Tensor): The split weight of each dual quad. If None, the algorithm
            will split based on minimum angle.
        aabb (list, tuple, np.ndarray, torch.Tensor): The axis-aligned bounding box of the mesh.
        voxel_size (float, list, tuple, np.ndarray, torch.Tensor): The size of each voxel.
        grid_size (int, list, tuple, np.ndarray, torch.Tensor): The size of the grid.
            NOTE: One of voxel_size and grid_size must be provided.
        train (bool): Whether to use training mode.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh.
        faces (torch.Tensor): The faces of the mesh.
    """
    device = coords.device
    if device not in _EDGE_NEIGHBOR_VOXEL_OFFSET:
        _EDGE_NEIGHBOR_VOXEL_OFFSET[device] = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # x-axis
                [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # y-axis
                [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # z-axis
            ],
            dtype=torch.int,
            device=device,
        ).unsqueeze(0)
    if device not in _QUAD_SPLIT_1:
        _QUAD_SPLIT_1[device] = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device, requires_grad=False)
    if device not in _QUAD_SPLIT_2:
        _QUAD_SPLIT_2[device] = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device, requires_grad=False)
    if device not in _QUAD_SPLIT_TRAIN:
        _QUAD_SPLIT_TRAIN[device] = torch.tensor([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=torch.long, device=device, requires_grad=False)
    edge_neighbor_voxel_offset = _EDGE_NEIGHBOR_VOXEL_OFFSET[device]
    quad_split_1 = _QUAD_SPLIT_1[device]
    quad_split_2 = _QUAD_SPLIT_2[device]
    quad_split_train = _QUAD_SPLIT_TRAIN[device]

    # AABB
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Voxel size
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
    assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
    assert voxel_size.dim() == 1, f"voxel_size must be a 1D tensor, but got {voxel_size.shape}"
    assert voxel_size.size(0) == 3, f"voxel_size must have 3 elements, but got {voxel_size.size(0)}"
    assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
    assert grid_size.dim() == 1, f"grid_size must be a 1D tensor, but got {grid_size.shape}"
    assert grid_size.size(0) == 3, f"grid_size must have 3 elements, but got {grid_size.size(0)}"

    # Extract mesh
    N = dual_vertices.shape[0]
    mesh_vertices = (coords.float() + dual_vertices) / (2 * N) - 0.5

    # Store active voxels into hashmap
    hashmap = _init_hashmap(grid_size, 2 * N, device=coords.device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1), *grid_size.tolist())

    # Find connected voxels
    edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + edge_neighbor_voxel_offset  # (N, 3, 4, 3)
    connected_voxel = edge_neighbor_voxel[intersected_flag]  # (M, 4, 3)
    M = connected_voxel.shape[0]
    connected_voxel_hash_key = torch.cat([torch.zeros((M * 4, 1), dtype=torch.int, device=coords.device), connected_voxel.reshape(-1, 3)], dim=1)
    connected_voxel_indices = _C.hashmap_lookup_3d_cuda(*hashmap, connected_voxel_hash_key, *grid_size.tolist()).reshape(M, 4).int()
    connected_voxel_valid = (connected_voxel_indices != 0xFFFFFFFF).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].int()  # (L, 4)
    L = quad_indices.shape[0]

    # Construct triangles
    if not train:
        mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)
        if split_weight is None:
            # if split 1
            atempt_triangles_0 = quad_indices[:, quad_split_1]
            normals0 = torch.cross(mesh_vertices[atempt_triangles_0[:, 1]] - mesh_vertices[atempt_triangles_0[:, 0]], mesh_vertices[atempt_triangles_0[:, 2]] - mesh_vertices[atempt_triangles_0[:, 0]])
            normals1 = torch.cross(mesh_vertices[atempt_triangles_0[:, 2]] - mesh_vertices[atempt_triangles_0[:, 1]], mesh_vertices[atempt_triangles_0[:, 3]] - mesh_vertices[atempt_triangles_0[:, 1]])
            align0 = (normals0 * normals1).sum(dim=1, keepdim=True).abs()
            # if split 2
            atempt_triangles_1 = quad_indices[:, quad_split_2]
            normals0 = torch.cross(mesh_vertices[atempt_triangles_1[:, 1]] - mesh_vertices[atempt_triangles_1[:, 0]], mesh_vertices[atempt_triangles_1[:, 2]] - mesh_vertices[atempt_triangles_1[:, 0]])
            normals1 = torch.cross(mesh_vertices[atempt_triangles_1[:, 2]] - mesh_vertices[atempt_triangles_1[:, 1]], mesh_vertices[atempt_triangles_1[:, 3]] - mesh_vertices[atempt_triangles_1[:, 1]])
            align1 = (normals0 * normals1).sum(dim=1, keepdim=True).abs()
            # select split
            mesh_triangles = torch.where(align0 > align1, atempt_triangles_0, atempt_triangles_1).reshape(-1, 3)
        else:
            split_weight_ws = split_weight[quad_indices]
            split_weight_ws_02 = split_weight_ws[:, 0] * split_weight_ws[:, 2]
            split_weight_ws_13 = split_weight_ws[:, 1] * split_weight_ws[:, 3]
            mesh_triangles = torch.where(split_weight_ws_02 > split_weight_ws_13, quad_indices[:, quad_split_1], quad_indices[:, quad_split_2]).reshape(-1, 3)
    else:
        assert split_weight is not None, "split_weight must be provided in training mode"
        mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)
        quad_vs = mesh_vertices[quad_indices]
        mean_v02 = (quad_vs[:, 0] + quad_vs[:, 2]) / 2
        mean_v13 = (quad_vs[:, 1] + quad_vs[:, 3]) / 2
        split_weight_ws = split_weight[quad_indices]
        split_weight_ws_02 = split_weight_ws[:, 0] * split_weight_ws[:, 2]
        split_weight_ws_13 = split_weight_ws[:, 1] * split_weight_ws[:, 3]
        mid_vertices = (split_weight_ws_02 * mean_v02 + split_weight_ws_13 * mean_v13) / (split_weight_ws_02 + split_weight_ws_13)
        mesh_vertices = torch.cat([mesh_vertices, mid_vertices], dim=0)
        quad_indices = torch.cat([quad_indices, torch.arange(N, N + L, device="cuda").unsqueeze(1)], dim=1)
        mesh_triangles = quad_indices[:, quad_split_train].reshape(-1, 3)

    return mesh_vertices, mesh_triangles
