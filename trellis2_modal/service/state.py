"""
State serialization for MeshWithVoxel.

Converts between GPU tensors and numpy arrays for network transfer.
Uses LZ4 compression on the client side (see client/compression.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trellis2.representations import MeshWithVoxel


def pack_state(mesh: MeshWithVoxel) -> dict[str, Any]:
    """
    Pack MeshWithVoxel into a serializable dictionary.

    All GPU tensors are moved to CPU and converted to numpy arrays.
    Slice objects in layout are converted to [start, stop] lists.

    Args:
        mesh: MeshWithVoxel from generation pipeline

    Returns:
        Dictionary with numpy arrays, ready for JSON serialization
    """
    return {
        "vertices": mesh.vertices.cpu().numpy(),
        "faces": mesh.faces.cpu().numpy(),
        "attrs": mesh.attrs.cpu().numpy(),
        "coords": mesh.coords.cpu().numpy(),
        "voxel_size": float(mesh.voxel_size),
        "voxel_shape": list(mesh.voxel_shape),
        "origin": mesh.origin.cpu().tolist(),
        "layout": {k: [v.start, v.stop] for k, v in mesh.layout.items()},
    }


def unpack_state(state: dict[str, Any]) -> MeshWithVoxel:
    """
    Reconstruct MeshWithVoxel from a packed state dictionary.

    All arrays are converted to CUDA tensors. This function assumes
    CUDA is available (it runs on the Modal GPU service).

    Args:
        state: Dictionary from pack_state()

    Returns:
        MeshWithVoxel with tensors on CUDA device

    Raises:
        RuntimeError: If CUDA is not available
    """
    import torch
    from trellis2.representations import MeshWithVoxel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. unpack_state() must run on GPU service."
        )

    layout = {k: slice(v[0], v[1]) for k, v in state["layout"].items()}

    return MeshWithVoxel(
        vertices=torch.tensor(state["vertices"], device="cuda", dtype=torch.float32),
        faces=torch.tensor(state["faces"], device="cuda", dtype=torch.int32),
        origin=state["origin"],
        voxel_size=state["voxel_size"],
        coords=torch.tensor(state["coords"], device="cuda"),
        attrs=torch.tensor(state["attrs"], device="cuda", dtype=torch.float32),
        voxel_shape=torch.Size(state["voxel_shape"]),
        layout=layout,
    )
