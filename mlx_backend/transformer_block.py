"""
Modulated Sparse Transformer Cross Block for MLX.
Implements AdaLN modulation → self-attn → cross-attn → FFN.
"""
import mlx.core as mx
import mlx.nn as nn
from .norm import LayerNorm32
from .attention import MlxMultiHeadAttention


class MlxSparseFeedForwardNet(nn.Module):
    """FFN with GELU activation."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = MlxSparseSequential(
            nn.Linear(channels, hidden),
            nn.GELU(approx="precise"),
            nn.Linear(hidden, channels),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


class MlxSparseSequential(nn.Module):
    """Sequential container that passes feats through layers."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MlxModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse transformer cross-attention block with AdaLN modulation.
    Matches PyTorch ModulatedSparseTransformerCrossBlock.

    With share_mod=True: uses per-block learnable `modulation` param + shared `mod` input.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        rope_freq: tuple = (1.0, 10000.0),
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.share_mod = share_mod

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)

        self.self_attn = MlxMultiHeadAttention(
            channels, num_heads,
            type="self", qkv_bias=qkv_bias,
            use_rope=use_rope, rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MlxMultiHeadAttention(
            channels, num_heads,
            ctx_channels=ctx_channels,
            type="cross", qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = MlxSparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)

        if not share_mod:
            self.adaLN_modulation = MlxSparseSequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )
        else:
            self.modulation = mx.zeros((6 * channels,))

    def __call__(
        self,
        x: mx.array,
        mod: mx.array,
        context: mx.array,
        rope_cache: tuple = None,
    ) -> mx.array:
        """
        Args:
            x: (N, C) sparse features
            mod: (1, 6*C) or (1, C) modulation signal from timestep
            context: (M, ctx_C) conditioning features
            rope_cache: (cos, sin) precomputed RoPE
        """
        if self.share_mod:
            mods = (self.modulation + mod).astype(mod.dtype)
        else:
            mods = self.adaLN_modulation(mod)

        # Split into 6 modulation signals: each (1, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            mx.split(mods, 6, axis=-1)

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, rope_cache=rope_cache)
        h = h * gate_msa
        x = x + h

        # Cross-attention (no AdaLN, just norm2 with affine)
        h = self.norm2(x)
        h = self.cross_attn(h, context=context)
        x = x + h

        # FFN with AdaLN
        h = self.norm3(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        return x


class MlxModulatedTransformerCrossBlock(nn.Module):
    """
    Dense transformer cross-attention block with AdaLN modulation.
    Used for SparseStructureFlowModel (dense 16^3 input).
    Supports batched input (B, N, C) for batched CFG.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        rope_freq: tuple = (1.0, 10000.0),
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.share_mod = share_mod

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)

        self.self_attn = MlxMultiHeadAttention(
            channels, num_heads,
            type="self", qkv_bias=qkv_bias,
            use_rope=use_rope, rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MlxMultiHeadAttention(
            channels, num_heads,
            ctx_channels=ctx_channels,
            type="cross", qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = MlxSparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)

        if not share_mod:
            self.adaLN_modulation = MlxSparseSequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )
        else:
            self.modulation = mx.zeros((6 * channels,))

    def __call__(
        self,
        x: mx.array,
        mod: mx.array,
        context: mx.array,
        rope_cache: tuple = None,
    ) -> mx.array:
        """
        Args:
            x: (N, C) or (B, N, C) dense features
            mod: (B, 6*C) modulation signal
            context: (M, ctx_C) or (B, M, ctx_C) conditioning features
            rope_cache: (cos, sin) precomputed RoPE
        """
        if self.share_mod:
            mods = (self.modulation + mod).astype(mod.dtype)
        else:
            mods = self.adaLN_modulation(mod)

        if x.ndim == 3:
            # Batched: mods is (B, 6*C), need (B, 1, C) for broadcasting
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                mx.split(mods, 6, axis=-1)
            shift_msa = shift_msa[:, None, :]
            scale_msa = scale_msa[:, None, :]
            gate_msa = gate_msa[:, None, :]
            shift_mlp = shift_mlp[:, None, :]
            scale_mlp = scale_mlp[:, None, :]
            gate_mlp = gate_mlp[:, None, :]
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                mx.split(mods, 6, axis=-1)

        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, rope_cache=rope_cache)
        h = h * gate_msa
        x = x + h

        h = self.norm2(x)
        h = self.cross_attn(h, context=context)
        x = x + h

        h = self.norm3(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        return x
