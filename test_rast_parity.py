"""
Rasterization parity test for Trellis2 texture baking.
Tests the exact UV-space rasterization scenario used by postprocess.py:
  - Positions are UV coords mapped to clip space (z=0, w=1)
  - All triangles share the same depth
  - Each face must get its correct unique ID

This is a focused unit test for the Metal rasterizer in the Trellis2 context.
For full E2E parity (Metal vs CPU postprocess), see test_postprocess_parity.py.
"""
import torch
import numpy as np
import struct
import pytest


def float_to_triidx(f):
    if f <= 16777216.0:
        return int(f)
    bits = struct.unpack('I', struct.pack('f', f))[0]
    return bits - 0x4a800000


@pytest.fixture
def mtl_rast():
    try:
        from mtldiffrast.torch.ops import MtlRasterizeContext, rasterize
        ctx = MtlRasterizeContext()
        return ctx, rasterize
    except ImportError:
        pytest.skip("mtldiffrast not built")


class TestTrellis2RastParity:
    """UV-space rasterization as used by Trellis2 postprocess.py."""

    def test_uv_quad_mesh(self, mtl_rast):
        """Simple UV quad — the minimal texture baking scenario."""
        ctx, rasterize = mtl_rast

        # UV coords mapped to clip space: uv * 2 - 1, z=0, w=1
        pos = torch.tensor([
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 0.0, 1.0],
            [ 1.0,  1.0, 0.0, 1.0],
            [-1.0,  1.0, 0.0, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)

        rast_out, _ = rasterize(ctx, pos, tri, resolution=[512, 512])
        rast_np = rast_out[0].numpy()

        # Full coverage
        covered = (rast_np[:, :, 3] != 0).sum()
        assert covered == 512 * 512, f"Expected full coverage, got {covered}"

        # Both face IDs present
        face_ids = set()
        for py in range(0, 512, 32):
            for px in range(0, 512, 32):
                if rast_np[py, px, 3] != 0:
                    fid = float_to_triidx(rast_np[py, px, 3]) - 1
                    face_ids.add(fid)
        assert face_ids == {0, 1}, f"Expected face IDs {{0, 1}}, got {face_ids}"

    def test_random_uv_mesh(self, mtl_rast):
        """Random UV layout with many triangles — simulates real Trellis2 mesh."""
        ctx, rasterize = mtl_rast
        torch.manual_seed(42)

        # Generate a grid mesh with random UV perturbation
        N = 20  # 20x20 grid = 800 triangles
        verts = []
        for y in range(N + 1):
            for x in range(N + 1):
                u = x / N
                v = y / N
                # Small random perturbation to avoid perfectly uniform spacing
                if 0 < x < N and 0 < y < N:
                    u += (torch.rand(1).item() - 0.5) * 0.02
                    v += (torch.rand(1).item() - 0.5) * 0.02
                cx = u * 2.0 - 1.0
                cy = v * 2.0 - 1.0
                verts.append([cx, cy, 0.0, 1.0])

        tris = []
        for y in range(N):
            for x in range(N):
                i = y * (N + 1) + x
                tris.append([i, i + 1, i + N + 2])
                tris.append([i, i + N + 2, i + N + 1])

        pos = torch.tensor(verts, dtype=torch.float32)
        tri = torch.tensor(tris, dtype=torch.int32)

        rast_out, _ = rasterize(ctx, pos, tri, resolution=[256, 256])
        rast_np = rast_out[0].numpy()

        # Should have high coverage
        covered = (rast_np[:, :, 3] != 0).sum()
        total = 256 * 256
        coverage_pct = covered / total * 100
        assert coverage_pct > 90, f"Expected >90% coverage, got {coverage_pct:.1f}%"

        # Should have many distinct face IDs (800 total)
        face_ids = set()
        for py in range(256):
            for px in range(256):
                if rast_np[py, px, 3] != 0:
                    fid = float_to_triidx(rast_np[py, px, 3]) - 1
                    face_ids.add(fid)
        # At 256x256 resolution, most of the 800 triangles should appear
        assert len(face_ids) > 600, f"Expected >600 distinct face IDs, got {len(face_ids)}"

    def test_barycentrics_interpolation_valid(self, mtl_rast):
        """Verify interpolated barycentrics are valid for texture sampling."""
        ctx, rasterize = mtl_rast
        from mtldiffrast.torch.ops import interpolate

        pos = torch.tensor([
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 0.0, 1.0],
            [ 1.0,  1.0, 0.0, 1.0],
            [-1.0,  1.0, 0.0, 1.0],
        ], dtype=torch.float32)
        tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
        # UV coordinates as vertex attributes
        uv_attr = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32)

        rast_out, _ = rasterize(ctx, pos, tri, resolution=[64, 64])
        interp_out, _ = interpolate(uv_attr, rast_out, tri)
        interp_np = interp_out[0].numpy()

        # Interpolated UVs should be in [0, 1]
        mask = rast_out[0, :, :, 3].numpy() != 0
        u_vals = interp_np[mask, 0]
        v_vals = interp_np[mask, 1]
        assert (u_vals >= -0.01).all() and (u_vals <= 1.01).all(), f"u out of range: [{u_vals.min()}, {u_vals.max()}]"
        assert (v_vals >= -0.01).all() and (v_vals <= 1.01).all(), f"v out of range: [{v_vals.min()}, {v_vals.max()}]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
