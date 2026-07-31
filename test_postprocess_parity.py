"""End-to-end parity checks for the Metal and CPU postprocess pipelines."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pytest
import torch

from o_voxel import postprocess, postprocess_cpu
from trellis2 import backends
from trellis2.mesh_integrity import measure


_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


def _probe_metal() -> tuple[bool, str]:
    """Require the complete, working Metal stack rather than import success."""

    if os.environ.get("TRELLIS_DISABLE_METAL", "0") == "1":
        return False, "Metal disabled by TRELLIS_DISABLE_METAL=1"

    postprocess_ready = bool(
        getattr(postprocess, "_HAS_DR", False)
        and getattr(postprocess, "_HAS_MESH", False)
        and getattr(postprocess, "_BACKEND", None) == "metal"
    )
    if not postprocess_ready:
        return False, "o_voxel.postprocess Metal rasterizer/mesh backend unavailable"

    backends_ready = bool(
        not getattr(backends, "METAL_DISABLED", False)
        and getattr(backends, "HAS_MPS", False)
        and getattr(backends, "_dr_backend", None) == "metal"
        and getattr(backends, "_mesh_backend", None) == "metal"
    )
    if not backends_ready:
        return False, "trellis2 Metal backend unavailable"

    try:
        probes = backends.probe_metal_backends()
    except (ImportError, RuntimeError, OSError) as exc:
        return False, f"trellis2 Metal probe failed: {exc}"

    failures = [
        f"{name}: {result.get('error', 'probe failed')}"
        for name, result in probes.items()
        if not result.get("ok", False)
    ]
    if failures:
        return False, "; ".join(failures)
    return True, ""


_METAL_AVAILABLE, _METAL_SKIP_REASON = _probe_metal()
requires_metal = pytest.mark.skipif(
    not _METAL_AVAILABLE,
    reason=_METAL_SKIP_REASON or "Metal backend unavailable",
)


def _subdivided_octahedron(
    subdivisions: int = 3,
    radius: float = 0.36,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic, outward-wound sphere-like closed triangle mesh."""

    vertices = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    vertices = [vertex * radius for vertex in vertices]
    faces = np.array(
        [
            [4, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ],
        dtype=np.int32,
    )

    for _ in range(subdivisions):
        midpoint_indices: dict[tuple[int, int], int] = {}
        next_faces: list[list[int]] = []

        def midpoint_index(left: int, right: int) -> int:
            edge = (left, right) if left < right else (right, left)
            if edge not in midpoint_indices:
                midpoint = (vertices[left] + vertices[right]) * 0.5
                midpoint *= radius / np.linalg.norm(midpoint)
                midpoint_indices[edge] = len(vertices)
                vertices.append(midpoint)
            return midpoint_indices[edge]

        for first, second, third in faces:
            first = int(first)
            second = int(second)
            third = int(third)
            first_second = midpoint_index(first, second)
            second_third = midpoint_index(second, third)
            third_first = midpoint_index(third, first)
            next_faces.extend(
                [
                    [first, first_second, third_first],
                    [second, second_third, first_second],
                    [third, third_first, second_third],
                    [first_second, second_third, third_first],
                ]
            )
        faces = np.asarray(next_faces, dtype=np.int32)

    vertices_array = np.asarray(vertices, dtype=np.float32)
    triangles = vertices_array[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    centers = triangles.mean(axis=1)
    inward = np.einsum("ij,ij->i", normals, centers) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return vertices_array, faces


def _synthetic_textured_input(grid_size: int = 32) -> dict[str, Any]:
    """Return a closed 512-face mesh and a small synthetic sparse PBR volume."""

    vertices, faces = _subdivided_octahedron()

    axis = np.arange(grid_size, dtype=np.int32)
    coords = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    normalized_coords = coords.astype(np.float32) / (grid_size - 1) * 2.0 - 1.0

    attr_volume = np.empty((coords.shape[0], 6), dtype=np.float32)
    attr_volume[:, :3] = np.array([0.25, 0.50, 0.75], dtype=np.float32)
    attr_volume[:, :3] += 0.015 * normalized_coords
    attr_volume[:, 3] = 0.15
    attr_volume[:, 4] = 0.65
    attr_volume[:, 5] = 1.0

    return {
        "vertices": torch.from_numpy(vertices),
        "faces": torch.from_numpy(faces),
        "attr_volume": torch.from_numpy(attr_volume),
        "coords": torch.from_numpy(coords),
        "attr_layout": dict(_ATTR_LAYOUT),
        "aabb": np.array([[-0.5] * 3, [0.5] * 3], dtype=np.float32),
        "grid_size": grid_size,
    }


def _export(
    exporter: Callable[..., Any],
    *,
    remesh: bool,
    texture_size: int = 256,
) -> Any:
    return exporter(
        **_synthetic_textured_input(),
        decimation_target=None,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1.0,
        remesh_project=0.0,
        verbose=False,
        use_tqdm=False,
    )


def _geometry_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(mesh.vertices),
        np.ascontiguousarray(mesh.faces),
    )


def _mean_base_color(mesh: Any) -> np.ndarray:
    image = mesh.visual.material.baseColorTexture
    pixels = np.asarray(image, dtype=np.float32)
    assert pixels.ndim == 3 and pixels.shape[2] >= 3
    return pixels[..., :3].mean(axis=(0, 1)) / 255.0


@requires_metal
def test_metal_stability() -> None:
    # Metal DC remesh is NOT bytewise deterministic: hashmap/atomic insertion
    # order varies run to run, so vertex/face counts drift slightly. The
    # production contract is metric-level: every run must pass the same
    # integrity gates and land on the same bounds.
    first = _export(postprocess.to_glb, remesh=True)
    second = _export(postprocess.to_glb, remesh=True)

    first_vertices, first_faces = _geometry_arrays(first)
    second_vertices, second_faces = _geometry_arrays(second)
    assert first_vertices.size > 0 and first_faces.size > 0

    face_drift = abs(first_faces.shape[0] - second_faces.shape[0]) / max(
        first_faces.shape[0], 1
    )
    assert face_drift <= 0.02, face_drift

    np.testing.assert_allclose(
        np.asarray(first.bounds), np.asarray(second.bounds), atol=1e-3
    )

    for metrics in (
        measure(first_vertices, first_faces),
        measure(second_vertices, second_faces),
    ):
        assert metrics["boundary_edges"] <= 8
        assert metrics["winding_consistent"] is True


@requires_metal
def test_metal_vs_cpu_loose_parity() -> None:
    metal = _export(postprocess.to_glb, remesh=True)
    # The kdtree/CPU exporter is a baking fallback and does not remesh.
    cpu = _export(postprocess_cpu.to_glb, remesh=False)

    metal_vertices, metal_faces = _geometry_arrays(metal)
    cpu_vertices, cpu_faces = _geometry_arrays(cpu)
    assert metal_vertices.size > 0 and metal_faces.size > 0
    assert cpu_vertices.size > 0 and cpu_faces.size > 0
    assert cpu_faces.shape[0] == _synthetic_textured_input()["faces"].shape[0]

    cpu_bounds = np.asarray(cpu.bounds)
    metal_bounds = np.asarray(metal.bounds)
    relative_bound_error = np.abs(metal_bounds - cpu_bounds) / np.maximum(
        np.abs(cpu_bounds),
        np.finfo(np.float32).eps,
    )
    # Narrow-band remesh wraps the surface at up to ~band voxels outward; on a
    # tiny synthetic fixture this inflates bounds by several percent. 12% keeps
    # the check meaningful (catches axis swaps/scale bugs) without fighting the
    # band envelope.
    assert np.all(relative_bound_error <= 0.12), relative_bound_error

    np.testing.assert_allclose(
        _mean_base_color(metal),
        _mean_base_color(cpu),
        rtol=0.0,
        atol=10.0 / 255.0,
    )


def test_cpu_path_warns_on_remesh() -> None:
    with pytest.warns(
        RuntimeWarning,
        match="kdtree/CPU path cannot remesh; candidate will not be promotable",
    ):
        result = _export(
            postprocess_cpu.to_glb,
            remesh=True,
            texture_size=64,
        )

    assert result.vertices.size > 0
    assert result.faces.shape[0] == _synthetic_textured_input()["faces"].shape[0]


@requires_metal
def test_remesh_output_passes_integrity_gate() -> None:
    result = _export(postprocess.to_glb, remesh=True)
    vertices, faces = _geometry_arrays(result)
    metrics = measure(vertices, faces)

    assert metrics["boundary_edges"] == 0
    assert metrics["winding_consistent"] is True
