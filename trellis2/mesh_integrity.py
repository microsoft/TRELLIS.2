"""Geometry-integrity measurements for triangle meshes and binary glTF files."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BIN_CHUNK = 0x004E4942
_GLB_MAGIC = b"glTF"

_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_INDEX_COMPONENT_TYPES = {5121, 5123, 5125}


def _component_labels(face_count: int, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return face-component labels using vectorized, monotonic union-find."""

    parents = np.arange(face_count, dtype=np.int64)
    if face_count == 0 or left.size == 0:
        return parents

    while True:
        while True:
            grandparents = parents[parents]
            if np.array_equal(grandparents, parents):
                break
            parents = grandparents

        left_roots = parents[left]
        right_roots = parents[right]
        different = left_roots != right_roots
        if not np.any(different):
            break

        higher = np.maximum(left_roots[different], right_roots[different])
        lower = np.minimum(left_roots[different], right_roots[different])
        np.minimum.at(parents, higher, lower)

    return parents


def measure(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Measure topology, winding, connectivity, and signed volume of a mesh."""

    raw_vertices = np.asarray(vertices)
    raw_faces = np.asarray(faces)
    if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if not np.issubdtype(raw_vertices.dtype, np.floating):
        raise TypeError("vertices must have a floating-point dtype")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise TypeError("faces must have an integer dtype")

    vertex_count = int(raw_vertices.shape[0])
    face_count = int(raw_faces.shape[0])
    if face_count:
        if np.issubdtype(raw_faces.dtype, np.signedinteger) and np.any(raw_faces < 0):
            raise ValueError("face indices must be non-negative")
        if np.any(raw_faces >= vertex_count):
            raise ValueError("face index is out of range")

    welded_vertices, inverse = np.unique(raw_vertices, axis=0, return_inverse=True)
    welded_faces = inverse[raw_faces.astype(np.int64, copy=False)]

    degenerate = (
        (welded_faces[:, 0] == welded_faces[:, 1])
        | (welded_faces[:, 1] == welded_faces[:, 2])
        | (welded_faces[:, 2] == welded_faces[:, 0])
    )
    degenerate_faces = int(np.count_nonzero(degenerate))
    nondegenerate_faces = welded_faces[~degenerate]

    if nondegenerate_faces.shape[0]:
        canonical_faces = np.sort(nondegenerate_faces, axis=1)
        unique_canonical, first_indices = np.unique(
            canonical_faces, axis=0, return_index=True
        )
        duplicate_faces = int(nondegenerate_faces.shape[0] - unique_canonical.shape[0])
        analyzed_faces = nondegenerate_faces[first_indices]
    else:
        duplicate_faces = 0
        analyzed_faces = np.empty((0, 3), dtype=np.int64)

    analyzed_count = int(analyzed_faces.shape[0])
    if analyzed_count:
        directed_edges = analyzed_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        undirected_edges = np.sort(directed_edges, axis=1)
        edge_order = np.lexsort((undirected_edges[:, 1], undirected_edges[:, 0]))
        sorted_edges = undirected_edges[edge_order]

        group_start = np.empty(sorted_edges.shape[0], dtype=bool)
        group_start[0] = True
        group_start[1:] = (
            (sorted_edges[1:, 0] != sorted_edges[:-1, 0])
            | (sorted_edges[1:, 1] != sorted_edges[:-1, 1])
        )
        group_indices = np.flatnonzero(group_start)
        edge_multiplicities = np.diff(
            np.append(group_indices, sorted_edges.shape[0])
        )

        boundary_edges = int(np.count_nonzero(edge_multiplicities == 1))
        non_manifold_edges = int(np.count_nonzero(edge_multiplicities > 2))

        paired_groups = group_indices[edge_multiplicities == 2]
        if paired_groups.size:
            first_half_edges = directed_edges[edge_order[paired_groups]]
            second_half_edges = directed_edges[edge_order[paired_groups + 1]]
            opposite = (
                (first_half_edges[:, 0] == second_half_edges[:, 1])
                & (first_half_edges[:, 1] == second_half_edges[:, 0])
            )
            winding_consistent = bool(np.all(opposite))
        else:
            winding_consistent = True

        same_as_previous = ~group_start[1:]
        ordered_face_indices = edge_order // 3
        adjacent_left = ordered_face_indices[:-1][same_as_previous]
        adjacent_right = ordered_face_indices[1:][same_as_previous]
        roots = _component_labels(analyzed_count, adjacent_left, adjacent_right)
        component_roots, component_ids = np.unique(roots, return_inverse=True)
        connected_components = int(component_roots.size)
        unique_edge_count = int(group_indices.size)
    else:
        boundary_edges = 0
        non_manifold_edges = 0
        winding_consistent = True
        connected_components = 0
        unique_edge_count = 0
        component_ids = np.empty(0, dtype=np.int64)

    if analyzed_count:
        volume_vertices = welded_vertices.astype(np.float64, copy=False)
        corners = volume_vertices[analyzed_faces]
        signed_face_volumes = np.einsum(
            "ij,ij->i",
            corners[:, 0],
            np.cross(corners[:, 1], corners[:, 2]),
        ) / 6.0
        component_volumes = np.bincount(
            component_ids,
            weights=signed_face_volumes,
            minlength=connected_components,
        )
        negative_volume_components = int(
            np.count_nonzero(component_volumes < -1e-12)
        )
        total_volume = float(component_volumes.sum(dtype=np.float64))
    else:
        negative_volume_components = 0
        total_volume = 0.0

    vertices_welded = int(welded_vertices.shape[0])
    return {
        "vertices_raw": vertex_count,
        "vertices_welded": vertices_welded,
        "faces_input": face_count,
        "faces_analyzed": analyzed_count,
        "degenerate_faces": degenerate_faces,
        "duplicate_faces": duplicate_faces,
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "winding_consistent": winding_consistent,
        "connected_components": connected_components,
        "negative_volume_components": negative_volume_components,
        "total_volume": total_volume,
        "watertight": boundary_edges == 0 and non_manifold_edges == 0,
        "euler_characteristic": vertices_welded - unique_edge_count + analyzed_count,
    }


def _load_glb(path: Path) -> tuple[dict[str, Any], memoryview]:
    payload = path.read_bytes()
    if len(payload) < 12:
        raise ValueError(f"truncated GLB: {path}")

    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != _GLB_MAGIC or version != 2 or declared_length != len(payload):
        raise ValueError(f"not a complete glTF 2.0 binary: {path}")

    payload_view = memoryview(payload)
    document: dict[str, Any] | None = None
    binary: memoryview | None = None
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError(f"truncated GLB chunk header: {path}")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(payload):
            raise ValueError(f"truncated GLB chunk: {path}")
        chunk = payload_view[offset:chunk_end]
        offset = chunk_end

        if chunk_type == _GLB_JSON_CHUNK:
            document = json.loads(
                bytes(chunk).rstrip(b" \t\r\n\0").decode("utf-8")
            )
        elif chunk_type == _GLB_BIN_CHUNK:
            if binary is not None:
                raise ValueError(f"GLB has multiple BIN chunks: {path}")
            binary = chunk

    if document is None:
        raise ValueError(f"missing GLB JSON chunk: {path}")
    if binary is None:
        binary = payload_view[len(payload) :]
    return document, binary


def _accessor_array(
    document: dict[str, Any], binary: memoryview, accessor_index: int
) -> np.ndarray:
    try:
        accessor = document["accessors"][accessor_index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"invalid accessor index: {accessor_index}") from exc

    if accessor.get("sparse") is not None:
        raise ValueError("sparse accessors are not supported")
    if "bufferView" not in accessor:
        raise ValueError("accessor has no bufferView")

    component_type = int(accessor.get("componentType", -1))
    accessor_type = accessor.get("type")
    if component_type not in _COMPONENT_DTYPES:
        raise ValueError(f"unsupported accessor componentType: {component_type}")
    if accessor_type not in _TYPE_COMPONENTS:
        raise ValueError(f"unsupported accessor type: {accessor_type}")

    count = int(accessor.get("count", -1))
    if count < 0:
        raise ValueError("accessor count must be non-negative")
    dtype = _COMPONENT_DTYPES[component_type]
    component_count = _TYPE_COMPONENTS[accessor_type]
    element_size = dtype.itemsize * component_count

    try:
        view = document["bufferViews"][int(accessor["bufferView"])]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("accessor has an invalid bufferView") from exc
    if int(view.get("buffer", 0)) != 0:
        raise ValueError("only the embedded GLB buffer is supported")

    view_offset = int(view.get("byteOffset", 0))
    view_length = int(view.get("byteLength", -1))
    accessor_offset = int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", element_size))
    if view_offset < 0 or view_length < 0 or accessor_offset < 0:
        raise ValueError("negative accessor or bufferView range")
    if stride < element_size:
        raise ValueError("bufferView byteStride is smaller than the accessor element")
    if view_offset + view_length > len(binary):
        raise ValueError("bufferView exceeds the GLB BIN chunk")

    required = accessor_offset
    if count:
        required += (count - 1) * stride + element_size
    if required > view_length:
        raise ValueError("accessor exceeds its bufferView")

    shape = (count,) if component_count == 1 else (count, component_count)
    if count == 0:
        return np.empty(shape, dtype=dtype)
    strides = (stride,) if component_count == 1 else (stride, dtype.itemsize)
    array = np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=binary,
        offset=view_offset + accessor_offset,
        strides=strides,
    )
    return np.array(array, copy=True)


def measure_glb(path: str) -> dict[str, Any]:
    """Parse a binary glTF file and measure all of its triangle primitives."""

    document, binary = _load_glb(Path(path))
    vertex_chunks: list[np.ndarray] = []
    face_chunks: list[np.ndarray] = []
    vertex_offset = 0

    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if int(primitive.get("mode", 4)) != 4:
                raise ValueError("only TRIANGLES mesh primitives are supported")
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                raise ValueError("mesh primitive has no POSITION accessor")

            position_index = int(attributes["POSITION"])
            position_accessor = document["accessors"][position_index]
            if position_accessor.get("type") != "VEC3":
                raise ValueError("POSITION accessor must have type VEC3")
            if int(position_accessor.get("componentType", -1)) != 5126:
                raise ValueError("POSITION accessor must use float32 components")
            positions = _accessor_array(document, binary, position_index)

            if "indices" in primitive:
                index_accessor_index = int(primitive["indices"])
                index_accessor = document["accessors"][index_accessor_index]
                component_type = int(index_accessor.get("componentType", -1))
                if index_accessor.get("type") != "SCALAR":
                    raise ValueError("indices accessor must have type SCALAR")
                if component_type not in _INDEX_COMPONENT_TYPES:
                    raise ValueError(
                        "indices accessor must use uint8, uint16, or uint32 components"
                    )
                indices = _accessor_array(
                    document, binary, index_accessor_index
                ).astype(np.int64, copy=False)
            else:
                indices = np.arange(positions.shape[0], dtype=np.int64)

            if indices.size % 3:
                raise ValueError("triangle primitive index count is not divisible by 3")
            if indices.size and int(indices.max()) >= positions.shape[0]:
                raise ValueError("primitive index is out of range")

            vertex_chunks.append(positions)
            face_chunks.append(indices.reshape(-1, 3) + vertex_offset)
            vertex_offset += int(positions.shape[0])

    if vertex_chunks:
        vertices = np.concatenate(vertex_chunks, axis=0)
        faces = np.concatenate(face_chunks, axis=0)
    else:
        vertices = np.empty((0, 3), dtype=np.float32)
        faces = np.empty((0, 3), dtype=np.int64)
    return measure(vertices, faces)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="binary glTF (.glb) file to measure")
    parser.add_argument("--json", action="store_true", help="print JSON")
    arguments = parser.parse_args()

    result = measure_glb(arguments.path)
    print(json.dumps(result) if arguments.json else result)


if __name__ == "__main__":
    _main()
