"""
Sparse spatial operations for MLX: Downsample, Upsample, Channel2Spatial, Spatial2Channel.
"""
import logging
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .sparse_tensor import MlxSparseTensor

logger = logging.getLogger(__name__)


class MlxSparseDownsample(nn.Module):
    """Downsample sparse tensor by factor, using mean pooling."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(self, x: MlxSparseTensor) -> MlxSparseTensor:
        t0 = time.time()
        N_in = x.feats.shape[0]

        cache = x.get_spatial_cache(f'downsample_{self.factor}')
        if cache is None:
            DIM = x.coords.shape[-1] - 1  # 3
            coords = x.coords  # (N, 4)
            batch_col = coords[:, 0]
            spatial_cols = [coords[:, i + 1] // self.factor for i in range(DIM)]

            MAX = [(int(x.spatial_shape[i]) + self.factor - 1) // self.factor for i in range(DIM)]
            OFFSET = [1] * (DIM + 1)
            for i in range(DIM - 1, -1, -1):
                OFFSET[i] = OFFSET[i + 1] * MAX[i]
            # batch offset
            batch_offset = OFFSET[0]

            code = batch_col.astype(mx.int32) * batch_offset
            for i in range(DIM):
                code = code + spatial_cols[i].astype(mx.int32) * OFFSET[i + 1]

            unique_codes, idx = mx.unique(code, return_inverse=True)

            # Reconstruct new coords from unique codes
            new_batch = unique_codes // batch_offset
            remainder = unique_codes % batch_offset
            new_spatial = []
            for i in range(DIM):
                new_spatial.append(remainder // OFFSET[i + 1])
                remainder = remainder % OFFSET[i + 1]
            new_coords = mx.stack([new_batch] + new_spatial, axis=-1).astype(mx.int32)
        else:
            new_coords, idx = cache

        # Scatter-mean: vectorized scatter-add + counts
        N_new = new_coords.shape[0]
        C = x.feats.shape[1]
        new_feats = _mlx_scatter_mean(x.feats, idx, N_new)

        out = MlxSparseTensor(new_feats, new_coords, shape=(x.shape[0], C))
        out._scale = tuple(s * self.factor for s in x._scale)
        out._spatial_cache = x._spatial_cache

        if cache is None:
            x.register_spatial_cache(f'downsample_{self.factor}', (new_coords, idx))
            out.register_spatial_cache(f'upsample_{self.factor}', (x.coords, idx))
            out.register_spatial_cache('shape', tuple(MAX))

        dt = time.time() - t0
        logger.debug("[MLX] Downsample: N %d → %d (factor=%d), %.2fs", N_in, N_new, self.factor, dt)
        return out


def _mlx_scatter_mean(feats: mx.array, idx: mx.array, n_out: int) -> mx.array:
    """
    Scatter-mean using vectorized scatter-add.
    feats: (N, C), idx: (N,) -> (n_out, C)
    """
    C = feats.shape[1]
    feats_f32 = feats.astype(mx.float32)

    # Scatter-add features and counts using .at[].add()
    sums = mx.zeros((n_out, C), dtype=mx.float32)
    counts = mx.zeros((n_out, 1), dtype=mx.float32)
    idx_int = idx.astype(mx.int32)
    sums = sums.at[idx_int].add(feats_f32)
    counts = counts.at[idx_int].add(mx.ones((feats.shape[0], 1), dtype=mx.float32))
    return (sums / mx.maximum(counts, 1.0)).astype(feats.dtype)


class MlxSparseUpsample(nn.Module):
    """Upsample sparse tensor by factor using nearest neighbor."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(self, x: MlxSparseTensor, subdivision: MlxSparseTensor = None) -> MlxSparseTensor:
        t0 = time.time()
        N_in = x.feats.shape[0]
        DIM = x.coords.shape[-1] - 1  # 3
        cache = x.get_spatial_cache(f'upsample_{self.factor}')

        if cache is None:
            if subdivision is None:
                raise ValueError('Cache not found. Provide subdivision tensor.')
            sub = subdivision.feats  # (N, factor^DIM)
            F_DIM = self.factor ** DIM

            # Vectorized: find active sub-voxels via numpy nonzero
            sub_flat = sub.reshape(-1)
            mx.eval(sub_flat)
            active_np = np.where(np.array(sub_flat) > 0)[0].astype(np.int32)
            active_indices = mx.array(active_np)

            parent_idx = active_indices // F_DIM
            subidx = active_indices % F_DIM

            parent_coords = x.coords[parent_idx].astype(mx.int32)
            batch_col = parent_coords[:, :1]
            spatial = parent_coords[:, 1:] * self.factor
            for d in range(DIM):
                offset = (subidx // (self.factor ** d) % self.factor).astype(mx.int32)[:, None]
                spatial = mx.concatenate([
                    spatial[:, :d],
                    spatial[:, d:d+1] + offset,
                    spatial[:, d+1:],
                ], axis=-1)
            new_coords = mx.concatenate([batch_col, spatial], axis=-1)
            idx = parent_idx
        else:
            new_coords, idx = cache

        new_feats = x.feats[idx]
        out = MlxSparseTensor(new_feats, new_coords, shape=(x.shape[0], x.feats.shape[1]))
        out._scale = tuple(s / self.factor for s in x._scale)
        if cache is not None:
            out._spatial_cache = x._spatial_cache

        dt = time.time() - t0
        N_out = new_feats.shape[0]
        logger.debug("[MLX] Upsample: N %d → %d (factor=%d), %.2fs", N_in, N_out, self.factor, dt)
        return out


class MlxSparseSpatial2Channel(nn.Module):
    """Downsample by rearranging spatial dims into channels."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(self, x: MlxSparseTensor) -> MlxSparseTensor:
        t0 = time.time()
        N_in = x.feats.shape[0]
        DIM = x.coords.shape[-1] - 1
        F = self.factor
        cache = x.get_spatial_cache(f'spatial2channel_{F}')

        if cache is None:
            coords = x.coords
            batch_col = coords[:, 0]
            spatial_cols = [coords[:, i + 1] // F for i in range(DIM)]

            # Sub-index within the factor cube
            subidx_parts = [coords[:, i + 1] % F for i in range(DIM)]
            subidx = subidx_parts[0].astype(mx.int32)
            for d in range(1, DIM):
                subidx = subidx + subidx_parts[d].astype(mx.int32) * (F ** d)

            MAX = [(int(x.spatial_shape[i]) + F - 1) // F for i in range(DIM)]
            OFFSET = [1] * (DIM + 1)
            for i in range(DIM - 1, -1, -1):
                OFFSET[i] = OFFSET[i + 1] * MAX[i]
            batch_offset = OFFSET[0]

            code = batch_col.astype(mx.int32) * batch_offset
            for i in range(DIM):
                code = code + spatial_cols[i].astype(mx.int32) * OFFSET[i + 1]

            unique_codes, idx = mx.unique(code, return_inverse=True)

            new_batch = unique_codes // batch_offset
            remainder = unique_codes % batch_offset
            new_spatial = []
            for i in range(DIM):
                new_spatial.append(remainder // OFFSET[i + 1])
                remainder = remainder % OFFSET[i + 1]
            new_coords = mx.stack([new_batch] + new_spatial, axis=-1).astype(mx.int32)
        else:
            new_coords, idx, subidx = cache

        # Pack features: scatter into (N_new * F^DIM, C) then reshape to (N_new, C * F^DIM)
        N_new = new_coords.shape[0]
        C = x.feats.shape[1]
        F_DIM = F ** DIM
        new_feats = mx.zeros((N_new * F_DIM, C), dtype=x.feats.dtype)
        flat_idx = idx * F_DIM + subidx.astype(mx.int32)
        new_feats = new_feats.at[flat_idx].add(x.feats)
        new_feats = new_feats.reshape(N_new, C * F_DIM)

        out = MlxSparseTensor(
            new_feats, new_coords,
            shape=(x.shape[0], C * F_DIM) if x.shape is not None else None,
        )
        out._scale = tuple(s * F for s in x._scale)
        out._spatial_cache = x._spatial_cache

        if cache is None:
            x.register_spatial_cache(f'spatial2channel_{F}', (new_coords, idx, subidx))
            out.register_spatial_cache(f'channel2spatial_{F}', (x.coords, idx, subidx))
            out.register_spatial_cache('shape', tuple(MAX))

        dt = time.time() - t0
        logger.debug("[MLX] Spatial2Channel: N %d → %d, C %d → %d, %.2fs",
                     N_in, N_new, C, C * F_DIM, dt)
        return out


class MlxSparseChannel2Spatial(nn.Module):
    """Upsample by rearranging channels into spatial dims."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(self, x: MlxSparseTensor, subdivision: MlxSparseTensor = None) -> MlxSparseTensor:
        t0 = time.time()
        N_in = x.feats.shape[0]
        DIM = x.coords.shape[-1] - 1
        F = self.factor
        F_DIM = F ** DIM
        cache = x.get_spatial_cache(f'channel2spatial_{F}')

        if cache is None:
            if subdivision is None:
                raise ValueError('Cache not found. Provide subdivision tensor.')
            sub = subdivision.feats  # (N, F_DIM)

            # Vectorized: find active sub-voxels via numpy nonzero
            sub_flat = sub.reshape(-1)
            mx.eval(sub_flat)
            active_np = np.where(np.array(sub_flat) > 0)[0].astype(np.int32)
            active_indices = mx.array(active_np)

            parent_idx = active_indices // F_DIM
            subidx = active_indices % F_DIM

            parent_coords = x.coords[parent_idx].astype(mx.int32)
            batch_col = parent_coords[:, :1]
            spatial = parent_coords[:, 1:] * F
            for d in range(DIM):
                offset = (subidx // (F ** d) % F).astype(mx.int32)[:, None]
                spatial = mx.concatenate([
                    spatial[:, :d],
                    spatial[:, d:d+1] + offset,
                    spatial[:, d+1:],
                ], axis=-1)
            new_coords = mx.concatenate([batch_col, spatial], axis=-1)
            idx = parent_idx
        else:
            new_coords, idx, subidx = cache

        # Unpack: reshape (N, C * F^DIM) -> (N * F^DIM, C), then gather
        C_packed = x.feats.shape[1]
        C = C_packed // F_DIM
        x_feats = x.feats.reshape(x.feats.shape[0] * F_DIM, C)
        flat_idx = idx * F_DIM + subidx.astype(mx.int32)
        new_feats = x_feats[flat_idx]

        out = MlxSparseTensor(
            new_feats, new_coords,
            shape=(x.shape[0], C) if x.shape is not None else None,
        )
        out._scale = tuple(s / F for s in x._scale)
        if cache is not None:
            out._spatial_cache = x._spatial_cache

        dt = time.time() - t0
        N_out = new_feats.shape[0]
        logger.debug("[MLX] Channel2Spatial: N %d → %d, C %d → %d, %.2fs",
                     N_in, N_out, C_packed, C, dt)
        return out
