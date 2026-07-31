"""
Flow matching models in MLX.
MlxSparseStructureFlowModel: dense 16^3 structure flow
MlxSLatFlowModel: sparse structured latent flow (shape + texture)
"""
import logging
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .norm import LayerNorm32
from .transformer_block import (
    MlxModulatedSparseTransformerCrossBlock,
    MlxModulatedTransformerCrossBlock,
    MlxSparseSequential,
)
from .rope import build_rope_freqs, compute_rope_phases
from .sparse_tensor import MlxSparseTensor, mlx_sparse_cat

logger = logging.getLogger(__name__)


def _metal_mem_mb() -> str:
    """Current Metal memory usage in MB."""
    try:
        active = mx.get_active_memory() / 1024**2
        peak = mx.get_peak_memory() / 1024**2
        return f"{active:.0f}MB (peak {peak:.0f}MB)"
    except Exception:
        return "N/A"


class MlxTimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = MlxSparseSequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self._timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

    @staticmethod
    def _timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding


class MlxSparseStructureFlowModel(nn.Module):
    """
    Dense flow model for sparse structure sampling.
    Input: (B, 8, 16, 16, 16) noise → (B, 8, 16, 16, 16) output.
    Internally flattened to (B, 4096, C) dense sequence.
    """

    def __init__(
        self,
        resolution: int = 16,
        in_channels: int = 8,
        model_channels: int = 1536,
        cond_channels: int = 1024,
        out_channels: int = 8,
        num_blocks: int = 30,
        num_heads: int = 12,
        mlp_ratio: float = 5.3334,
        pe_mode: str = "rope",
        share_mod: bool = True,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.pe_mode = pe_mode
        self.share_mod = share_mod

        self.t_embedder = MlxTimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = MlxSparseSequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        self.input_layer = nn.Linear(in_channels, model_channels)

        self.blocks = [
            MlxModulatedTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ]

        self.out_layer = nn.Linear(model_channels, out_channels)

        # Precompute RoPE for 16^3 grid
        if pe_mode == "rope":
            head_dim = model_channels // num_heads
            freqs = build_rope_freqs(head_dim, dim=3)
            coords_np = np.stack(np.meshgrid(
                np.arange(resolution), np.arange(resolution), np.arange(resolution),
                indexing='ij'
            ), axis=-1).reshape(-1, 3)
            coords_mx = mx.array(coords_np.astype(np.int32))
            self._rope_cos, self._rope_sin = compute_rope_phases(coords_mx, freqs, head_dim)

        self._compiled_blocks = None

    def _run_blocks(self, h, t_emb, cond, rope_cos, rope_sin):
        rope_cache = (rope_cos, rope_sin)
        for block in self.blocks:
            h = block(h, t_emb, cond, rope_cache=rope_cache)
        return h

    def _run_blocks_no_rope(self, h, t_emb, cond):
        for block in self.blocks:
            h = block(h, t_emb, cond, rope_cache=None)
        return h

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        B = x.shape[0]
        R = self.resolution

        logger.debug("[MLX] StructureFlow: x=%s dtype=%s, mem=%s",
                     x.shape, x.dtype, _metal_mem_mb())

        # Flatten: (B, C, D, H, W) -> (B, D*H*W, C)
        h = x.reshape(B, self.in_channels, -1).transpose(0, 2, 1)
        h = self.input_layer(h)

        # Match upstream bfloat16 reduced-precision casting (manual_cast)
        compute_dtype = mx.bfloat16
        h = h.astype(compute_dtype)

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.astype(compute_dtype)
        cond = cond.astype(compute_dtype)

        if self.pe_mode == "rope":
            if self._compiled_blocks is None:
                try:
                    self._compiled_blocks = mx.compile(self._run_blocks)
                    logger.info("[MLX] StructureFlow: using mx.compile for block loop")
                except Exception as e:
                    logger.warning("[MLX] StructureFlow: mx.compile failed (%s), falling back to per-block eval", e)
                    self._compiled_blocks = False
            if self._compiled_blocks:
                h = self._compiled_blocks(h, t_emb, cond, self._rope_cos, self._rope_sin)
            else:
                rope_cache = (self._rope_cos, self._rope_sin)
                for i, block in enumerate(self.blocks):
                    h = block(h, t_emb, cond, rope_cache=rope_cache)
        else:
            if self._compiled_blocks is None:
                try:
                    self._compiled_blocks = mx.compile(self._run_blocks_no_rope)
                    logger.info("[MLX] StructureFlow: using mx.compile for block loop (no rope)")
                except Exception as e:
                    logger.warning("[MLX] StructureFlow: mx.compile failed (%s), falling back to per-block eval", e)
                    self._compiled_blocks = False
            if self._compiled_blocks:
                h = self._compiled_blocks(h, t_emb, cond)
            else:
                for block in self.blocks:
                    h = block(h, t_emb, cond, rope_cache=None)

        logger.debug("[MLX] StructureFlow blocks done, mem=%s", _metal_mem_mb())

        # Cast back to float32 before final norm (matches upstream cast back to input dtype)
        h = h.astype(mx.float32)
        # Two-pass LayerNorm (matches LayerNorm32 / upstream F.layer_norm precision)
        mean = mx.mean(h, axis=-1, keepdims=True)
        h = h - mean
        var = mx.mean(h * h, axis=-1, keepdims=True)
        h = h * mx.rsqrt(var + 1e-5)
        h = self.out_layer(h)

        h = h.transpose(0, 2, 1).reshape(B, self.out_channels, R, R, R)
        return h


class MlxSLatFlowModel(nn.Module):
    """
    Sparse structured latent flow model.
    Input: MlxSparseTensor with (N, in_channels) features.
    """

    def __init__(
        self,
        resolution: int = 64,
        in_channels: int = 32,
        model_channels: int = 1536,
        cond_channels: int = 1024,
        out_channels: int = 32,
        num_blocks: int = 30,
        num_heads: int = 12,
        mlp_ratio: float = 5.3334,
        pe_mode: str = "rope",
        share_mod: bool = True,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.pe_mode = pe_mode
        self.share_mod = share_mod

        self.t_embedder = MlxTimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = MlxSparseSequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        self.input_layer = nn.Linear(in_channels, model_channels)

        self.blocks = [
            MlxModulatedSparseTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ]

        self.out_layer = nn.Linear(model_channels, out_channels)

        if pe_mode == "rope":
            head_dim = model_channels // num_heads
            self._rope_freqs = build_rope_freqs(head_dim, dim=3)

        self._compiled_blocks = None

    def _run_blocks(self, h, t_emb, cond, rope_cos, rope_sin):
        rope_cache = (rope_cos, rope_sin)
        for block in self.blocks:
            h = block(h, t_emb, cond, rope_cache=rope_cache)
        return h

    def _run_blocks_no_rope(self, h, t_emb, cond):
        for block in self.blocks:
            h = block(h, t_emb, cond, rope_cache=None)
        return h

    def __call__(
        self,
        x: MlxSparseTensor,
        t: mx.array,
        cond: mx.array,
        concat_cond: MlxSparseTensor = None,
    ) -> MlxSparseTensor:
        if concat_cond is not None:
            x = mlx_sparse_cat([x, concat_cond], dim=-1)

        N = x.feats.shape[0]
        logger.debug("[MLX] SLatFlow: N=%d, in_ch=%d, dtype=%s, mem=%s",
                     N, x.feats.shape[1], x.feats.dtype, _metal_mem_mb())

        h = self.input_layer(x.feats)

        # Match upstream bfloat16 reduced-precision casting (manual_cast)
        compute_dtype = mx.bfloat16
        h = h.astype(compute_dtype)

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.astype(compute_dtype)
        cond = cond.astype(compute_dtype)

        # Compute RoPE from sparse coords
        if self.pe_mode == "rope":
            coords_3d = x.coords[:, 1:]
            rope_cos, rope_sin = compute_rope_phases(
                coords_3d, self._rope_freqs,
                self.model_channels // self.num_heads,
            )

            if self._compiled_blocks is None:
                try:
                    self._compiled_blocks = mx.compile(self._run_blocks)
                    logger.info("[MLX] SLatFlow: using mx.compile for block loop")
                except Exception as e:
                    logger.warning("[MLX] SLatFlow: mx.compile failed (%s), falling back to per-block eval", e)
                    self._compiled_blocks = False
            if self._compiled_blocks:
                h = self._compiled_blocks(h, t_emb, cond, rope_cos, rope_sin)
            else:
                rope_cache = (rope_cos, rope_sin)
                for block in self.blocks:
                    h = block(h, t_emb, cond, rope_cache=rope_cache)
        else:
            if self._compiled_blocks is None:
                try:
                    self._compiled_blocks = mx.compile(self._run_blocks_no_rope)
                    logger.info("[MLX] SLatFlow: using mx.compile for block loop (no rope)")
                except Exception as e:
                    logger.warning("[MLX] SLatFlow: mx.compile failed (%s), falling back to per-block eval", e)
                    self._compiled_blocks = False
            if self._compiled_blocks:
                h = self._compiled_blocks(h, t_emb, cond)
            else:
                for block in self.blocks:
                    h = block(h, t_emb, cond, rope_cache=None)

        logger.debug("[MLX] SLatFlow blocks done, N=%d, mem=%s", N, _metal_mem_mb())

        # Cast back to float32 before final norm (matches upstream cast back to input dtype)
        h = h.astype(mx.float32)
        # Two-pass LayerNorm (matches LayerNorm32 / upstream F.layer_norm precision)
        mean = mx.mean(h, axis=-1, keepdims=True)
        h = h - mean
        var = mx.mean(h * h, axis=-1, keepdims=True)
        h = h * mx.rsqrt(var + 1e-5)
        h = self.out_layer(h)
        return x.replace(h)
