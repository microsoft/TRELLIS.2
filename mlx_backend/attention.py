"""
Multi-head attention for MLX backend.
Fused scaled dot-product attention via Metal kernel, with QK-RMSNorm and RoPE.

Supports two modes:
- Sparse (unbatched): x is (N, C) — used for sparse flow models
- Dense (batched): x is (B, N, C) — used for dense structure flow with batched CFG
"""
import mlx.core as mx
import mlx.nn as nn
from .norm import SparseMultiHeadRMSNorm
from .rope import build_rope_freqs, compute_rope_phases, apply_rope


class MlxMultiHeadAttention(nn.Module):
    """
    Multi-head attention matching the PyTorch SparseMultiHeadAttention.

    For self-attention: fused QKV projection, optional QK-RMSNorm, optional RoPE.
    For cross-attention: separate Q and KV projections.

    Supports both (N, C) unbatched and (B, N, C) batched input.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: int = None,
        type: str = "self",
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: tuple = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels or channels
        self._type = type
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self._rope_freqs = build_rope_freqs(self.head_dim, dim=3, rope_freq=rope_freq)

    def __call__(
        self,
        x: mx.array,
        context: mx.array = None,
        rope_cache: tuple = None,
    ) -> mx.array:
        """
        Args:
            x: (N, C) or (B, N, C) input features
            context: (M, ctx_C) or (B, M, ctx_C) cross-attention context
            rope_cache: (cos, sin) precomputed RoPE phases

        Returns:
            Same shape as x
        """
        H = self.num_heads
        D = self.head_dim
        batched = x.ndim == 3

        if self._type == "self":
            qkv = self.to_qkv(x)  # (..., 3*C)
            if batched:
                B, N, _ = qkv.shape
                qkv = qkv.reshape(B, N, 3, H, D)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, H, D)
            else:
                qkv = qkv.reshape(-1, 3, H, D)
                q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (N, H, D)

            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)

            if self.use_rope and rope_cache is not None:
                cos, sin = rope_cache
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)

            out = self._sdpa(q, k, v, batched)
        else:
            q = self.to_q(x)
            kv = self.to_kv(context)
            if batched:
                B, N, _ = q.shape
                q = q.reshape(B, N, H, D)
                M = kv.shape[1]
                kv = kv.reshape(B, M, 2, H, D)
                k, v = kv[:, :, 0], kv[:, :, 1]  # (B, M, H, D)
            else:
                q = q.reshape(-1, H, D)
                kv = kv.reshape(-1, 2, H, D)
                k, v = kv[:, 0], kv[:, 1]

            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)

            out = self._sdpa(q, k, v, batched)

        if batched:
            out = out.reshape(B, -1, self.channels)
        else:
            out = out.reshape(-1, self.channels)
        return self.to_out(out)

    def _sdpa(self, q: mx.array, k: mx.array, v: mx.array, batched: bool = False) -> mx.array:
        """
        Fused scaled dot-product attention via Metal kernel.
        Inputs: (N, H, D) unbatched or (B, N, H, D) batched.
        """
        scale = self.head_dim ** -0.5
        if batched:
            # (B, N, H, D) -> (B, H, N, D)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            return out.transpose(0, 2, 1, 3)  # (B, N, H, D)
        else:
            # (N, H, D) -> (1, H, N, D)
            q = q.transpose(1, 0, 2)[None]
            k = k.transpose(1, 0, 2)[None]
            v = v.transpose(1, 0, 2)[None]
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            return out[0].transpose(1, 0, 2)  # (N, H, D)
