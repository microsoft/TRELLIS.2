"""Unified 3D grid sampling — flex_gemm on Metal/CUDA, F.grid_sample fallback."""
import torch
import torch.nn.functional as F

try:
    from flex_gemm.ops.grid_sample import grid_sample_3d as _flex_grid_sample
    _HAS_FLEX_GEMM = True
except ImportError:
    _HAS_FLEX_GEMM = False


def grid_sample_3d(feats, coords, shape, grid, mode='trilinear'):
    """Drop-in replacement for flex_gemm.ops.grid_sample.grid_sample_3d.

    Args:
        feats: [N, C] sparse voxel features
        coords: [N, 4] voxel coordinates (batch_idx, x, y, z)
        shape: torch.Size([B, C, D, H, W]) — sparse tensor shape
        grid: [B, M, 3] query points in voxel space
        mode: 'trilinear' (maps to F.grid_sample 'bilinear')
    Returns:
        [B*C, M] sampled features (matching flex_gemm output shape)
    """
    if _HAS_FLEX_GEMM:
        return _flex_grid_sample(feats, coords, shape, grid, mode=mode)

    # Dense volume fallback
    B, C = shape[0], shape[1]
    D, H, W = shape[2], shape[3], shape[4]
    device = feats.device

    dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=device)
    batch_idx = coords[:, 0].long()
    cx, cy, cz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
    dense_vol[batch_idx, :, cx, cy, cz] = feats

    # Normalize grid to [-1, 1] for F.grid_sample (expects z,y,x order)
    grid_norm = torch.stack([
        grid[..., 2] / (W - 1) * 2 - 1,
        grid[..., 1] / (H - 1) * 2 - 1,
        grid[..., 0] / (D - 1) * 2 - 1,
    ], dim=-1)
    grid_norm = grid_norm.reshape(B, 1, 1, -1, 3)

    sampled = F.grid_sample(dense_vol, grid_norm, mode='bilinear',
                            align_corners=True, padding_mode='border')
    # sampled: [B, C, 1, 1, M] -> reshape to match flex_gemm output
    M = grid.shape[1]
    return sampled.reshape(B * C, M)
