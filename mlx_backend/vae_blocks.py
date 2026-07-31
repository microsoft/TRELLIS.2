"""
VAE decoder building blocks in MLX.
Matches PyTorch SparseResBlock3d, SparseConvNeXtBlock3d, etc.
"""
import logging
import time
import mlx.core as mx
import mlx.nn as nn
from .norm import LayerNorm32
from .sparse_tensor import MlxSparseTensor
from .sparse_conv import MlxSparseConv3d
from .sparse_ops import (
    MlxSparseDownsample, MlxSparseUpsample,
    MlxSparseSpatial2Channel, MlxSparseChannel2Spatial,
)

logger = logging.getLogger(__name__)


class MlxSparseLinear(nn.Module):
    """Linear layer operating on sparse tensor features."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def __call__(self, x: MlxSparseTensor) -> MlxSparseTensor:
        return x.replace(self.linear(x.feats))


class MlxSparseConvNeXtBlock3d(nn.Module):
    """ConvNeXt-style block: conv → norm → MLP residual."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = MlxSparseConv3d(channels, channels, 3)
        hidden = int(channels * mlp_ratio)
        self.mlp = MlxConvNeXtMLP(channels, hidden)

    def __call__(self, x: MlxSparseTensor) -> MlxSparseTensor:
        t0 = time.time()
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        h = h.replace(self.mlp(h.feats))
        result = MlxSparseTensor(
            feats=h.feats + x.feats,
            coords=x.coords,
            shape=x.shape,
            _scale=x._scale,
            _spatial_cache=x._spatial_cache,
        )
        dt = time.time() - t0
        if dt > 0.5:
            logger.debug("[MLX] ConvNeXtBlock: N=%d, ch=%d, %.2fs",
                         x.feats.shape[0], self.channels, dt)
        return result


class MlxConvNeXtMLP(nn.Module):
    """MLP for ConvNeXt block: Linear → SiLU → Linear (zero-init)."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        # layers list used by weight remapping (mlp.layers.0/2)
        self.layers = [
            nn.Linear(channels, hidden),
            None,  # SiLU (not a module)
            nn.Linear(hidden, channels),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers[2](nn.silu(self.layers[0](x)))


class MlxSparseResBlockC2S3d(nn.Module):
    """
    Residual block with Channel2Spatial upsampling.
    Used in shape decoder to go from coarse to fine resolution.
    """

    def __init__(self, channels: int, out_channels: int = None,
                 pred_subdiv: bool = True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pred_subdiv = pred_subdiv

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        # conv1 outputs out_channels * 8 (spatial expansion)
        self.conv1 = MlxSparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = MlxSparseConv3d(self.out_channels, self.out_channels, 3)
        if pred_subdiv:
            self.to_subdiv = MlxSparseLinear(channels, 8)
        self.updown = MlxSparseChannel2Spatial(2)

    def _skip_connection(self, x: MlxSparseTensor) -> MlxSparseTensor:
        """Repeat features to match out_channels after C2S."""
        # x after C2S has channels // 8 features
        feats = x.feats
        c_in = feats.shape[1]
        repeats = self.out_channels // c_in
        if repeats > 1:
            feats = mx.repeat(feats, repeats, axis=1)
        return x.replace(feats[:, :self.out_channels])

    def __call__(self, x: MlxSparseTensor, subdiv: MlxSparseTensor = None):
        t0 = time.time()
        N_in = x.feats.shape[0]

        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)

        h = x.replace(self.norm1(x.feats))
        h = h.replace(nn.silu(h.feats))
        h = self.conv1(h)

        subdiv_bin = subdiv.replace((subdiv.feats > 0).astype(subdiv.feats.dtype)) if subdiv is not None else None
        h = self.updown(h, subdiv_bin)
        x = self.updown(x, subdiv_bin)

        h = h.replace(self.norm2(h.feats))
        h = h.replace(nn.silu(h.feats))
        h = self.conv2(h)

        skip = self._skip_connection(x)
        result = MlxSparseTensor(
            feats=h.feats + skip.feats,
            coords=h.coords,
            shape=h.shape,
            _scale=h._scale,
            _spatial_cache=h._spatial_cache,
        )

        dt = time.time() - t0
        logger.debug("[MLX] ResBlockC2S: N %d → %d, ch %d → %d, %.2fs",
                     N_in, result.feats.shape[0], self.channels, self.out_channels, dt)

        if self.pred_subdiv:
            return result, subdiv
        return result


class MlxSparseResBlockUpsample3d(nn.Module):
    """Residual block with nearest-neighbor upsampling."""

    def __init__(self, channels: int, out_channels: int = None,
                 pred_subdiv: bool = True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pred_subdiv = pred_subdiv

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = MlxSparseConv3d(channels, self.out_channels, 3)
        self.conv2 = MlxSparseConv3d(self.out_channels, self.out_channels, 3)
        if channels != self.out_channels:
            self.skip_connection = MlxSparseLinear(channels, self.out_channels)
        else:
            self.skip_connection = None
        if pred_subdiv:
            self.to_subdiv = MlxSparseLinear(channels, 8)
        self.updown = MlxSparseUpsample(2)

    def __call__(self, x: MlxSparseTensor, subdiv: MlxSparseTensor = None):
        t0 = time.time()
        N_in = x.feats.shape[0]

        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)

        h = x.replace(self.norm1(x.feats))
        h = h.replace(nn.silu(h.feats))
        subdiv_bin = subdiv.replace((subdiv.feats > 0).astype(subdiv.feats.dtype)) if subdiv is not None else None
        h = self.updown(h, subdiv_bin)
        x_up = self.updown(x, subdiv_bin)

        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(nn.silu(h.feats))
        h = self.conv2(h)

        if self.skip_connection is not None:
            skip = self.skip_connection(x_up)
        else:
            skip = x_up

        result = MlxSparseTensor(
            feats=h.feats + skip.feats,
            coords=h.coords,
            shape=h.shape,
            _scale=h._scale,
            _spatial_cache=h._spatial_cache,
        )

        dt = time.time() - t0
        logger.debug("[MLX] ResBlockUpsample: N %d → %d, ch %d → %d, %.2fs",
                     N_in, result.feats.shape[0], self.channels, self.out_channels, dt)

        if self.pred_subdiv:
            return result, subdiv
        return result
