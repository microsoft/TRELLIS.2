"""
Normalization layers for MLX backend.
Uses two-pass LayerNorm for parity with PyTorch (avoids fused kernel precision drift).
"""
import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.Module):
    """LayerNorm with two-pass variance for PyTorch parity.

    mx.fast.layer_norm uses single-pass parallel variance that drifts ~1.4e-6
    per call vs PyTorch's two-pass (~9.5e-7). Over 90+ norms per forward pass
    and 50 sampler steps, this compounds significantly. Manual two-pass stays
    in MLX's lazy graph while matching PyTorch numerics.
    """

    def __init__(self, dims: int, elementwise_affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        x_dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        var = mx.mean(centered * centered, axis=-1, keepdims=True)
        x = centered * mx.rsqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x.astype(x_dtype)


class SparseMultiHeadRMSNorm(nn.Module):
    """Per-head RMSNorm using mx.fast.rms_norm (fused Metal kernel).

    Equivalent to: L2_normalize(x) * gamma * sqrt(D)
    Which equals: rms_norm(x) * gamma (the sqrt(D) cancels).
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** 0.5
        self.gamma = mx.ones((heads, dim))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (..., H, D) — supports (N, H, D) or (B, N, H, D)."""
        x_dtype = x.dtype
        orig_shape = x.shape
        D = orig_shape[-1]
        # Flatten all dims except last for fused rms_norm
        x_flat = x.reshape(-1, D).astype(mx.float32)
        x_flat = mx.fast.rms_norm(x_flat, None, 1e-6)
        x = x_flat.reshape(orig_shape) * self.gamma
        return x.astype(x_dtype)
