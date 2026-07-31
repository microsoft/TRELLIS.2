"""
Sparse structure decoder in MLX — dense 3D conv network.
Converts flow output (1, 8, 16, 16, 16) → binary voxel grid (1, 1, 64, 64, 64).
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def conv3d(x: mx.array, weight: mx.array, bias: mx.array = None, padding: int = 1) -> mx.array:
    """3D convolution via mx.conv_general. x: (B,D,H,W,Ci), weight: (Co,Ci,kD,kH,kW)."""
    # MLX conv_general expects (B, spatial..., Ci) and weight (Co, kD, kH, kW, Ci)
    # PyTorch weight format: (Co, Ci, kD, kH, kW) → MLX: (Co, kD, kH, kW, Ci)
    w = weight.transpose(0, 2, 3, 4, 1)  # (Co, kD, kH, kW, Ci)
    out = mx.conv_general(x, w, padding=padding)
    if bias is not None:
        out = out + bias
    return out


class ChannelLayerNorm3d(nn.Module):
    """
    Channel LayerNorm for 3D data in channels-last format.
    Matches PT ChannelLayerNorm32: LayerNorm applied to the channel dimension.
    Input: (B, D, H, W, C) — normalizes over C.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        x_dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        var = mx.mean(centered * centered, axis=-1, keepdims=True)
        x = centered * mx.rsqrt(var + 1e-5) * self.weight + self.bias
        return x.astype(x_dtype)


class ResBlock3d(nn.Module):
    """ResBlock: ChannelLayerNorm → SiLU → Conv3d → ChannelLayerNorm → SiLU → Conv3d + skip.

    Uses ChannelLayerNorm to match PT's default norm_type="layer" (ChannelLayerNorm32).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm1 = ChannelLayerNorm3d(channels)
        self.norm2 = ChannelLayerNorm3d(channels)
        # Conv weights stored in PyTorch format (Co, Ci, kD, kH, kW)
        self.conv1 = {"weight": mx.zeros((channels, channels, 3, 3, 3)),
                      "bias": mx.zeros((channels,))}
        self.conv2 = {"weight": mx.zeros((channels, channels, 3, 3, 3)),
                      "bias": mx.zeros((channels,))}

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        h = self.norm1(x).astype(dtype)
        h = nn.silu(h)
        h = conv3d(h, self.conv1["weight"], self.conv1["bias"])
        h = self.norm2(h).astype(dtype)
        h = nn.silu(h)
        h = conv3d(h, self.conv2["weight"], self.conv2["bias"])
        return h + x


class PixelShuffleUpsample3d(nn.Module):
    """Conv3d then 3D pixel shuffle (factor 2): channels/8 at 2x resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Output channels = out_channels * 8 (for 2^3 pixel shuffle)
        self.out_channels = out_channels
        self.conv = {"weight": mx.zeros((out_channels * 8, in_channels, 3, 3, 3)),
                     "bias": mx.zeros((out_channels * 8,))}

    def __call__(self, x: mx.array) -> mx.array:
        h = conv3d(x, self.conv["weight"], self.conv["bias"])
        # Pixel shuffle 3D: (B, D, H, W, C*8) → (B, D*2, H*2, W*2, C)
        B, D, H, W, C8 = h.shape
        C = self.out_channels
        h = h.reshape(B, D, H, W, C, 2, 2, 2)
        h = h.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (B, D, 2, H, 2, W, 2, C)
        h = h.reshape(B, D * 2, H * 2, W * 2, C)
        return h


class MlxSparseStructureDecoder(nn.Module):
    """
    Dense 3D conv decoder for sparse structure prediction.
    Input: (B, latent_channels, D, H, W) — typically (1, 8, 16, 16, 16)
    Output: (B, 1, D', H', W') — binary occupancy grid
    """

    def __init__(self, out_channels: int = 1, latent_channels: int = 8,
                 num_res_blocks: int = 2, num_res_blocks_middle: int = 2,
                 channels: list = None, use_fp16: bool = True):
        super().__init__()
        if channels is None:
            channels = [512, 128, 32]

        self.use_fp16 = use_fp16

        # Input layer
        self.input_layer = {"weight": mx.zeros((channels[0], latent_channels, 3, 3, 3)),
                            "bias": mx.zeros((channels[0],))}

        # Middle blocks
        self.middle_block = [ResBlock3d(channels[0]) for _ in range(num_res_blocks_middle)]

        # Decoder blocks: [ResBlock × num_res_blocks, PixelShuffleUpsample] × stages
        self.blocks = []
        for i in range(len(channels)):
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock3d(channels[i]))
            if i < len(channels) - 1:
                self.blocks.append(PixelShuffleUpsample3d(channels[i], channels[i + 1]))

        # Output layer: ChannelLayerNorm → SiLU → Conv3d
        final_ch = channels[-1]
        self.out_layer = [
            ChannelLayerNorm3d(final_ch),
            None,  # SiLU
            {"weight": mx.zeros((out_channels, final_ch, 3, 3, 3)),
             "bias": mx.zeros((out_channels,))},
        ]

        self._compiled_forward = None

    def _run_blocks(self, h):
        """Middle + decoder blocks — compilable dense graph."""
        for block in self.middle_block:
            h = block(h)
        for block in self.blocks:
            h = block(h)
        return h

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, C, D, H, W) in PyTorch format → convert to (B, D, H, W, C) for MLX."""
        # Convert from (B, C, D, H, W) to (B, D, H, W, C)
        h = x.transpose(0, 2, 3, 4, 1)

        if self.use_fp16:
            h = h.astype(mx.float16)

        # Input conv
        h = conv3d(h, self.input_layer["weight"], self.input_layer["bias"])

        # Middle + decoder blocks (compiled)
        if self._compiled_forward is None:
            try:
                self._compiled_forward = mx.compile(self._run_blocks)
            except Exception:
                self._compiled_forward = False
        if self._compiled_forward:
            h = self._compiled_forward(h)
        else:
            h = self._run_blocks(h)

        # Output
        h = h.astype(mx.float32)
        h = self.out_layer[0](h)  # GroupNorm
        h = nn.silu(h)
        h = conv3d(h, self.out_layer[2]["weight"], self.out_layer[2]["bias"])

        # Convert back to (B, C, D, H, W)
        h = h.transpose(0, 4, 1, 2, 3)
        return h


def load_structure_decoder(model_path: str) -> MlxSparseStructureDecoder:
    """Load structure decoder from config + safetensors."""
    import json

    with open(f"{model_path}.json") as f:
        config = json.load(f)

    args = config['args']
    model = MlxSparseStructureDecoder(
        out_channels=args['out_channels'],
        latent_channels=args['latent_channels'],
        num_res_blocks=args['num_res_blocks'],
        num_res_blocks_middle=args['num_res_blocks_middle'],
        channels=args['channels'],
        use_fp16=args.get('use_fp16', True),
    )

    weights = mx.load(f"{model_path}.safetensors")

    # Remap weight keys for MLX module structure
    remapped = _remap_structure_decoder_weights(weights)
    model.load_weights(list(remapped.items()))
    return model


def _remap_structure_decoder_weights(weights: dict) -> dict:
    """Remap PyTorch weight keys to MLX module paths."""
    import re
    remapped = {}
    for k, v in weights.items():
        new_k = k
        # middle_block.N.xxx → middle_block.N.xxx (list indexing, no .layers.)
        # blocks.N.xxx → blocks.N.xxx
        # out_layer.0.xxx → out_layer.0.xxx (GroupNorm)
        # out_layer.2.xxx → out_layer.2.xxx (Conv)

        # For dict-based conv layers, map conv1.weight → conv1.weight etc.
        # These are already in the right format for our dict-based storage
        remapped[new_k] = v
    return remapped
