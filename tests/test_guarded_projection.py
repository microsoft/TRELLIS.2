import os

os.environ.setdefault("TRELLIS_DISABLE_METAL", "1")

import torch

from o_voxel.postprocess import _guarded_project_back


def _closest_point_on_triangle(point, triangle):
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = torch.dot(ab, ap)
    d2 = torch.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a, point.new_tensor([1.0, 0.0, 0.0])

    bp = point - b
    d3 = torch.dot(ab, bp)
    d4 = torch.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b, point.new_tensor([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        barycentric = torch.stack([1 - v, v, v.new_zeros(())])
        return a + v * ab, barycentric

    cp = point - c
    d5 = torch.dot(ab, cp)
    d6 = torch.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c, point.new_tensor([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        barycentric = torch.stack([1 - w, w.new_zeros(()), w])
        return a + w * ac, barycentric

    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        barycentric = torch.stack([w.new_zeros(()), 1 - w, w])
        return b + w * (c - b), barycentric

    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    barycentric = torch.stack([1 - v - w, v, w])
    return a + ab * v + ac * w, barycentric


class FakeBVH:
    """CPU-only brute-force closest-point stand-in for the Metal BVH."""

    def __init__(self, vertices, faces):
        self.triangles = vertices[faces.long()]

    def unsigned_distance(self, query, return_uvw=False):
        assert return_uvw
        distances = []
        face_ids = []
        barycentrics = []
        for point in query:
            best_distance_sq = None
            best_face_id = None
            best_barycentric = None
            for face_id, triangle in enumerate(self.triangles):
                closest, barycentric = _closest_point_on_triangle(point, triangle)
                distance_sq = torch.dot(point - closest, point - closest)
                if best_distance_sq is None or distance_sq < best_distance_sq:
                    best_distance_sq = distance_sq
                    best_face_id = face_id
                    best_barycentric = barycentric
            distances.append(best_distance_sq.sqrt())
            face_ids.append(best_face_id)
            barycentrics.append(best_barycentric)
        return (
            torch.stack(distances),
            torch.tensor(face_ids, dtype=torch.long, device=query.device),
            torch.stack(barycentrics),
        )


def _plane(extent=2.0):
    vertices = torch.tensor(
        [
            [-extent, -extent, 0.0],
            [extent, -extent, 0.0],
            [extent, extent, 0.0],
            [-extent, extent, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    return vertices, faces


def test_clean_plane_projects_all_dc_vertices():
    src_vertices, src_faces = _plane()
    dc_vertices = torch.tensor(
        [
            [-1.0, -1.0, 0.1],
            [1.0, -1.0, 0.1],
            [1.0, 1.0, 0.1],
            [-1.0, 1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    dc_faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)

    projected, moved_count, reverted_count = _guarded_project_back(
        dc_vertices,
        dc_faces,
        src_vertices,
        src_faces,
        FakeBVH(src_vertices, src_faces),
        strength=1.0,
        voxel_size=0.1,
    )

    torch.testing.assert_close(projected[:, :2], dc_vertices[:, :2])
    torch.testing.assert_close(projected[:, 2], torch.zeros(4))
    assert moved_count == len(dc_vertices)
    assert reverted_count == 0


def test_perpendicular_flap_is_rejected_by_normal_guard():
    plane_vertices, plane_faces = _plane()
    flap_vertices = torch.tensor(
        [
            [1.0, -2.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 1.0],
            [1.0, -2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    src_vertices = torch.cat([plane_vertices, flap_vertices], dim=0)
    src_faces = torch.cat(
        [
            plane_faces,
            torch.tensor([[4, 5, 6], [4, 6, 7]], dtype=torch.int32),
        ],
        dim=0,
    )
    dc_vertices = torch.tensor(
        [
            [-1.0, -1.0, 0.1],
            [0.95, -1.0, 0.1],
            [0.95, 1.0, 0.1],
            [-1.0, 1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    dc_faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)

    projected, moved_count, reverted_count = _guarded_project_back(
        dc_vertices,
        dc_faces,
        src_vertices,
        src_faces,
        FakeBVH(src_vertices, src_faces),
        strength=1.0,
        voxel_size=0.1,
    )

    torch.testing.assert_close(projected[[0, 3], 2], torch.zeros(2))
    torch.testing.assert_close(projected[[1, 2]], dc_vertices[[1, 2]])
    assert moved_count == 2
    assert reverted_count == 0


def test_distance_guard_rejects_far_dc_vertex():
    src_vertices, src_faces = _plane(extent=10.0)
    dc_vertices = torch.tensor(
        [[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    dc_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    projected, moved_count, reverted_count = _guarded_project_back(
        dc_vertices,
        dc_faces,
        src_vertices,
        src_faces,
        FakeBVH(src_vertices, src_faces),
        strength=1.0,
        voxel_size=1.0,
        max_dist_voxels=0.5,
        min_normal_agreement=0.0,
    )

    torch.testing.assert_close(projected[:2, 2], torch.zeros(2))
    torch.testing.assert_close(projected[2], dc_vertices[2])
    assert moved_count == 2
    assert reverted_count == 0


def test_projection_that_flips_face_is_reverted():
    src_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
        dtype=torch.float32,
    )
    src_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    dc_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    dc_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    bvh = FakeBVH(src_vertices, src_faces)

    _, face_id, uvw = bvh.unsigned_distance(dc_vertices, return_uvw=True)
    closest = (
        src_vertices[src_faces[face_id.long()].long()] * uvw.unsqueeze(-1)
    ).sum(dim=1)
    naive = dc_vertices + 2.0 * (closest - dc_vertices)
    normal_before = torch.cross(
        dc_vertices[1] - dc_vertices[0],
        dc_vertices[2] - dc_vertices[0],
        dim=0,
    )
    normal_naive = torch.cross(
        naive[1] - naive[0],
        naive[2] - naive[0],
        dim=0,
    )
    assert torch.dot(normal_before, normal_naive) < 0

    projected, moved_count, reverted_count = _guarded_project_back(
        dc_vertices,
        dc_faces,
        src_vertices,
        src_faces,
        bvh,
        strength=2.0,
        voxel_size=1.0,
        max_dist_voxels=2.0,
    )

    returned_normal = torch.cross(
        projected[1] - projected[0],
        projected[2] - projected[0],
        dim=0,
    )
    assert torch.dot(normal_before, returned_normal) >= 0
    torch.testing.assert_close(projected, dc_vertices)
    assert moved_count == 0
    assert reverted_count > 0
