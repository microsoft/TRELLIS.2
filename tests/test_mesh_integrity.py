import numpy as np

from trellis2.mesh_integrity import measure


def _cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_closed_cube() -> None:
    vertices, faces = _cube()

    result = measure(vertices, faces)

    assert result["watertight"] is True
    assert result["winding_consistent"] is True
    assert result["boundary_edges"] == 0
    assert result["non_manifold_edges"] == 0
    assert result["connected_components"] == 1
    assert result["negative_volume_components"] == 0


def test_cube_with_one_reversed_face() -> None:
    vertices, faces = _cube()
    faces[0] = faces[0, ::-1]

    result = measure(vertices, faces)

    assert result["winding_consistent"] is False
    assert result["boundary_edges"] == 0


def test_open_box_has_four_boundary_edges() -> None:
    vertices, faces = _cube()
    faces = np.delete(faces, [2, 3], axis=0)

    result = measure(vertices, faces)

    assert result["faces_input"] == 10
    assert result["boundary_edges"] == 4


def test_three_triangles_sharing_an_edge() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)

    result = measure(vertices, faces)

    assert result["non_manifold_edges"] == 1


def test_uv_split_cube_welds_before_edge_analysis() -> None:
    cube_vertices, _ = _cube()
    quads = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    vertices = np.concatenate([cube_vertices[quad] for quad in quads], axis=0)
    faces = np.concatenate(
        [
            np.array([[base, base + 1, base + 2], [base, base + 2, base + 3]])
            for base in range(0, 24, 4)
        ],
        axis=0,
    )

    result = measure(vertices, faces)

    assert result["vertices_raw"] == 24
    assert result["vertices_welded"] == 8
    assert result["boundary_edges"] == 0
    assert result["winding_consistent"] is True


def test_fully_inverted_cube_has_negative_volume() -> None:
    vertices, faces = _cube()

    result = measure(vertices, faces[:, ::-1])

    assert result["negative_volume_components"] == 1


def test_two_disjoint_cubes_have_two_components() -> None:
    vertices, faces = _cube()
    second_vertices = vertices + np.array([4.0, 0.0, 0.0])
    combined_vertices = np.concatenate([vertices, second_vertices], axis=0)
    combined_faces = np.concatenate([faces, faces + vertices.shape[0]], axis=0)

    result = measure(combined_vertices, combined_faces)

    assert result["connected_components"] == 2
