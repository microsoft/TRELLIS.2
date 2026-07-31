"""
Submanifold sparse 3D convolution for MLX.
Port of conv_pytorch.py algorithm: hash → neighbor map → gather → bmm → scatter.
"""
import itertools
import logging
import time
import mlx.core as mx
import mlx.nn as nn
from .sparse_tensor import MlxSparseTensor

logger = logging.getLogger(__name__)


def _build_neighbor_map(
    coords: mx.array,
    batch_size: int,
    spatial_shape: tuple,
    kernel_size: tuple,
    dilation: tuple,
) -> mx.array:
    """
    Build neighbor map for submanifold sparse conv.

    Returns:
        neighbor_map: (K, N) int32. neighbor_map[k, i] = index of neighbor, or N (pad index).
    """
    N = coords.shape[0]
    D, H, W = spatial_shape
    DHW = D * H * W

    # Build lookup table: flat_coord -> index
    flat_keys = (coords[:, 0].astype(mx.int32) * DHW +
                 coords[:, 1].astype(mx.int32) * (H * W) +
                 coords[:, 2].astype(mx.int32) * W +
                 coords[:, 3].astype(mx.int32))

    table_size = batch_size * DHW
    # Initialize lookup with N (= pad index)
    lookup = mx.full((table_size,), N, dtype=mx.int32)
    lookup = lookup.at[flat_keys].add(mx.arange(N, dtype=mx.int32) - N)

    # Generate kernel offsets
    kd, kh, kw = kernel_size
    dd, dh, dw = dilation
    offsets = []
    for dx, dy, dz in itertools.product(
        range(-(kd // 2), kd // 2 + 1),
        range(-(kh // 2), kh // 2 + 1),
        range(-(kw // 2), kw // 2 + 1),
    ):
        offsets.append((dx * dd, dy * dh, dz * dw))
    K = len(offsets)

    # For each offset, shift coords and lookup
    neighbor_maps = []
    for dx, dy, dz in offsets:
        sx = coords[:, 1].astype(mx.int32) + dx
        sy = coords[:, 2].astype(mx.int32) + dy
        sz = coords[:, 3].astype(mx.int32) + dz

        # Bounds check
        valid = ((sx >= 0) & (sx < D) &
                 (sy >= 0) & (sy < H) &
                 (sz >= 0) & (sz < W))

        flat_shifted = (coords[:, 0].astype(mx.int32) * DHW +
                        sx * (H * W) + sy * W + sz)
        # Clamp for safe indexing
        flat_shifted = mx.clip(flat_shifted, 0, table_size - 1)
        looked_up = lookup[flat_shifted]
        # Set invalid to N (pad index)
        looked_up = mx.where(valid, looked_up, N)
        neighbor_maps.append(looked_up)

    return mx.stack(neighbor_maps, axis=0)  # (K, N)


class MlxSparseConv3d(nn.Module):
    """
    Submanifold sparse 3D convolution in MLX.

    Weight format: (Co, Kd, Kh, Kw, Ci) — same as PyTorch checkpoint.
    At forward time, reshaped to (K, Ci, Co) for gather-matmul pattern.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * 3
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(dilation, int):
            self.dilation = (dilation,) * 3
        else:
            self.dilation = tuple(dilation)

        K = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        # Weight stored as (Co, Kd, Kh, Kw, Ci) for checkpoint compatibility
        self.weight = mx.zeros((out_channels, *self.kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: MlxSparseTensor) -> MlxSparseTensor:
        t0 = time.time()
        N = x.feats.shape[0]
        Co = self.out_channels
        Ci = self.in_channels

        # Get or build neighbor map
        cache_key = f'SubMConv3d_neighbor_cache_mlx_{self.kernel_size}_dilation{self.dilation}'
        neighbor_map = x.get_spatial_cache(cache_key)

        if neighbor_map is None:
            batch_size = x.shape[0]
            spatial_shape = x.spatial_shape
            neighbor_map = _build_neighbor_map(
                x.coords, batch_size, spatial_shape,
                self.kernel_size, self.dilation,
            )
            mx.eval(neighbor_map)  # eval now to free the O(D³) lookup table
            x.register_spatial_cache(cache_key, neighbor_map)

        K = neighbor_map.shape[0]

        # Reshape weight: (Co, Kd, Kh, Kw, Ci) -> (K, Ci, Co)
        w = self.weight.reshape(Co, K, Ci)  # (Co, K, Ci)
        w = w.transpose(1, 2, 0)  # (K, Ci, Co)

        # Pad feats with zero row for out-of-bounds indices
        # feats_padded[N] is zeros, so invalid neighbor lookups produce zero
        # from matmul naturally — no valid_mask needed.
        feats_padded = mx.concatenate([
            x.feats,
            mx.zeros((1, Ci), dtype=x.feats.dtype)
        ], axis=0)  # (N+1, Ci)

        # Target 512MB chunks — MLX handles chunked graphs well and prevents OOM
        max_bytes = 512 * 1024**2
        elem_size = 2 if w.dtype in (mx.float16, mx.bfloat16) else 4
        per_offset_bytes = N * max(Ci, Co) * elem_size * 2  # gather + matmul output
        chunk_size = max(1, min(K, int(max_bytes / max(per_offset_bytes, 1))))

        logger.debug("[MLX] SparseConv3d: N=%d, Ci=%d, Co=%d, K=%d, chunk_size=%d",
                     N, Ci, Co, K, chunk_size)

        result = mx.zeros((N, Co), dtype=w.dtype)
        if chunk_size >= K:
            # Single pass — no eval needed, keep graph lazy for caller
            gathered = feats_padded[neighbor_map]  # (K, N, Ci)
            out = mx.matmul(gathered.astype(w.dtype), w)  # (K, N, Co)
            result = mx.sum(out, axis=0)  # (N, Co)
        else:
            for start in range(0, K, chunk_size):
                end = min(start + chunk_size, K)
                nmap_chunk = neighbor_map[start:end]  # (chunk, N)
                gathered = feats_padded[nmap_chunk]  # (chunk, N, Ci)
                out = mx.matmul(gathered.astype(w.dtype), w[start:end])  # (chunk, N, Co)
                result = result + mx.sum(out, axis=0)  # (N, Co)
                mx.eval(result)  # free chunk intermediates between iterations

        result = result + self.bias

        dt = time.time() - t0
        if dt > 0.1:
            logger.debug("[MLX] SparseConv3d done: N=%d, %dx%d→%d, %.2fs", N, Ci, K, Co, dt)

        return x.replace(result.astype(x.feats.dtype))
