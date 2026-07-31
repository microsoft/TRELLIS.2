"""
VAE decoders in MLX: SparseUnetVaeDecoder and FlexiDualGridVaeDecoder.
"""
import logging
import time
import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Tuple
from .norm import LayerNorm32
from .sparse_tensor import MlxSparseTensor
from .sparse_conv import MlxSparseConv3d
from .vae_blocks import (
    MlxSparseLinear,
    MlxSparseConvNeXtBlock3d,
    MlxSparseResBlockC2S3d,
    MlxSparseResBlockUpsample3d,
)

logger = logging.getLogger(__name__)


class MlxSparseUnetVaeDecoder(nn.Module):
    """
    Sparse UNet VAE decoder.
    Matches PyTorch SparseUnetVaeDecoder architecture.
    """

    def __init__(
        self,
        out_channels: int,
        model_channels: list,
        latent_channels: int,
        num_blocks: list,
        block_type: list,
        up_block_type: list,
        block_args: list = None,
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.pred_subdiv = pred_subdiv
        self.use_fp16 = use_fp16

        self.output_layer = MlxSparseLinear(model_channels[-1], out_channels)
        self.from_latent = MlxSparseLinear(latent_channels, model_channels[0])

        # Build blocks
        BLOCK_TYPES = {
            'SparseConvNeXtBlock3d': MlxSparseConvNeXtBlock3d,
        }
        UP_BLOCK_TYPES = {
            'SparseResBlockC2S3d': MlxSparseResBlockC2S3d,
            'SparseResBlockUpsample3d': MlxSparseResBlockUpsample3d,
        }

        self.blocks = []
        for i in range(len(num_blocks)):
            stage = []
            BlockClass = BLOCK_TYPES[block_type[i]]
            for j in range(num_blocks[i]):
                stage.append(BlockClass(model_channels[i]))
            if i < len(num_blocks) - 1:
                UpBlockClass = UP_BLOCK_TYPES[up_block_type[i]]
                stage.append(UpBlockClass(
                    model_channels[i], model_channels[i + 1],
                    pred_subdiv=pred_subdiv,
                ))
            self.blocks.append(stage)

    def __call__(
        self,
        x: MlxSparseTensor,
        guide_subs: list = None,
        return_subs: bool = False,
    ):
        logger.info("[MLX] VAE decoder forward: N=%d, latent_ch=%d", x.feats.shape[0], x.feats.shape[1])
        t_total = time.time()

        h = self.from_latent(x)
        if self.use_fp16:
            h = h.replace(h.feats.astype(mx.float16))

        subs = []
        for i, stage in enumerate(self.blocks):
            t_stage = time.time()
            for j, block in enumerate(stage):
                if i < len(self.blocks) - 1 and j == len(stage) - 1:
                    # Up block
                    if self.pred_subdiv:
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)

            # Eval once per stage (not per block) to bound memory
            mx.eval(h.feats)
            dt_stage = time.time() - t_stage
            logger.info("[MLX] VAE stage %d/%d: N=%d, channels=%d, %.2fs",
                        i + 1, len(self.blocks), h.feats.shape[0], h.feats.shape[1], dt_stage)

        h = h.replace(h.feats.astype(x.feats.dtype))

        # Final layer norm (two-pass for parity)
        feats = h.feats.astype(mx.float32)
        mean = mx.mean(feats, axis=-1, keepdims=True)
        centered = feats - mean
        var = mx.mean(centered * centered, axis=-1, keepdims=True)
        feats = centered * mx.rsqrt(var + 1e-5)
        h = h.replace(feats.astype(x.feats.dtype))

        h = self.output_layer(h)

        dt_total = time.time() - t_total
        logger.info("[MLX] VAE decoder done: N=%d, out_ch=%d, total %.2fs",
                    h.feats.shape[0], h.feats.shape[1], dt_total)

        if return_subs:
            return h, subs
        return h

    def upsample(self, x: MlxSparseTensor, upsample_times: int) -> mx.array:
        """Run decoder up to upsample_times stages, return upsampled coords."""
        logger.info("[MLX] VAE upsample: N=%d, upsample_times=%d", x.feats.shape[0], upsample_times)
        h = self.from_latent(x)
        if self.use_fp16:
            h = h.replace(h.feats.astype(mx.float16))

        for i, stage in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(stage):
                if i < len(self.blocks) - 1 and j == len(stage) - 1:
                    h, sub = block(h)
                else:
                    h = block(h)
                mx.eval(h.feats)

        return h.coords


class MlxFlexiDualGridVaeDecoder(nn.Module):
    """
    FlexiDualGrid VAE decoder — wraps SparseUnetVaeDecoder.
    Outputs mesh vertices + intersection logits + quad_lerp.
    """

    def __init__(
        self,
        resolution: int,
        model_channels: list,
        latent_channels: int,
        num_blocks: list,
        block_type: list,
        up_block_type: list,
        block_args: list = None,
        use_fp16: bool = False,
        voxel_margin: float = 0.5,
    ):
        super().__init__()
        self.resolution = resolution
        self.voxel_margin = voxel_margin

        # out_channels = 7 (3 vertex + 3 intersection + 1 quad_lerp)
        self.decoder = MlxSparseUnetVaeDecoder(
            out_channels=7,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            block_args=block_args,
            use_fp16=use_fp16,
            pred_subdiv=True,
        )

    def set_resolution(self, resolution: int):
        self.resolution = resolution

    def __call__(self, x: MlxSparseTensor, return_subs: bool = False, **kwargs):
        decoded = self.decoder(x, return_subs=return_subs, **kwargs)

        if return_subs:
            h, subs = decoded
        else:
            h = decoded
            subs = None

        # Post-process: vertices = sigmoid(h[:, :3]), intersected = h[:, 3:6] > 0
        feats = h.feats
        vertices_feats = (1 + 2 * self.voxel_margin) * mx.sigmoid(feats[:, :3]) - self.voxel_margin
        intersected_feats = (feats[:, 3:6] > 0).astype(feats.dtype)
        quad_lerp_feats = mx.where(feats[:, 6:7] > 0, feats[:, 6:7], mx.zeros_like(feats[:, 6:7])) + mx.log1p(mx.exp(-mx.abs(feats[:, 6:7])))  # softplus

        result = (h, vertices_feats, intersected_feats, quad_lerp_feats)

        if return_subs:
            return result, subs
        return result

    def upsample(self, x: MlxSparseTensor, upsample_times: int) -> mx.array:
        return self.decoder.upsample(x, upsample_times)
