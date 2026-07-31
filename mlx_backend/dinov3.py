"""
DINOv3 ViT feature extractor in MLX.
Uses HuggingFace transformers weights but runs purely in MLX.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from typing import Optional


class MlxDINOv3PatchEmbed(nn.Module):
    """Patch embedding: Conv2d(3, dim, kernel=16, stride=16) + CLS + registers."""

    def __init__(self, dim: int = 1024, patch_size: int = 16, num_register_tokens: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.weight = mx.zeros((dim, patch_size, patch_size, 3))
        self.bias = mx.zeros((dim,))
        self.cls_token = mx.zeros((1, 1, dim))
        self.register_tokens = mx.zeros((1, num_register_tokens, dim))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, 3) -> (B, num_patches + 1 + num_reg, dim)"""
        B, H, W, C = x.shape
        P = self.patch_size
        nH, nW = H // P, W // P

        # Manual convolution via reshape + matmul (stride == kernel_size)
        x = x.reshape(B, nH, P, nW, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nH, nW, P, P, C)
        x = x.reshape(B, nH * nW, P * P * C)

        w = self.weight.reshape(self.weight.shape[0], -1).T  # (P*P*3, dim)
        patches = x @ w + self.bias  # (B, N, dim)

        cls = mx.broadcast_to(self.cls_token, (B, 1, patches.shape[-1]))
        reg = mx.broadcast_to(self.register_tokens, (B, self.num_register_tokens, patches.shape[-1]))
        return mx.concatenate([cls, reg, patches], axis=1)


class MlxDINOv3TransformerBlock(nn.Module):
    """ViT transformer block with layer scaling."""

    def __init__(self, dim: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MlxDINOv3Attention(dim, num_heads)
        hidden = int(dim * mlp_ratio)
        self.mlp = MlxDINOv3MLP(dim, hidden)
        # Layer scaling
        self.layer_scale1 = mx.ones((dim,))
        self.layer_scale2 = mx.ones((dim,))

    def __call__(self, x: mx.array, rope_cos: mx.array = None, rope_sin: mx.array = None,
                 num_prefix_tokens: int = 5) -> mx.array:
        h = self.norm1(x)
        h = self.attn(h, rope_cos, rope_sin, num_prefix_tokens=num_prefix_tokens)
        x = x + self.layer_scale1 * h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.layer_scale2 * h
        return x


class MlxDINOv3Attention(nn.Module):
    """Multi-head self-attention with RoPE (matching HF DINOv3ViTAttention).

    Uses separate Q, K, V projections (not fused QKV) to match HF weight layout.
    RoPE is applied only to patch tokens, skipping prefix (CLS + register) tokens.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=False)  # DINOv3: key has no bias
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, rope_cos: mx.array = None, rope_sin: mx.array = None,
                 num_prefix_tokens: int = 5) -> mx.array:
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)  # (B, H, N, D)
        k = self.k_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)

        if rope_cos is not None:
            # Apply RoPE only to patch tokens, skip prefix (CLS + registers)
            num_patches = rope_cos.shape[-2]
            q_prefix = q[:, :, :num_prefix_tokens]
            k_prefix = k[:, :, :num_prefix_tokens]
            q_patches = q[:, :, num_prefix_tokens:]
            k_patches = k[:, :, num_prefix_tokens:]

            q_patches = self._apply_rope(q_patches, rope_cos, rope_sin)
            k_patches = self._apply_rope(k_patches, rope_cos, rope_sin)

            q = mx.concatenate([q_prefix, q_patches], axis=2)
            k = mx.concatenate([k_prefix, k_patches], axis=2)

        scale = D ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.o_proj(out)

    @staticmethod
    def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Apply RoPE via rotate_half: x*cos + rotate_half(x)*sin.

        cos/sin: (1, N_patches, head_dim) — full head_dim, already tiled.
        x: (B, H, N_patches, head_dim).
        """
        # rotate_half: [-x2, x1]
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        x_rot = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + x_rot * sin


class MlxDINOv3MLP(nn.Module):
    """MLP with GELU activation."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class MlxDINOv3FeatureExtractor(nn.Module):
    """
    DINOv3 ViT-L/16 feature extractor in pure MLX.
    24 layers, dim=1024, 16 heads, patch_size=16.
    """

    def __init__(self, dim: int = 1024, num_heads: int = 16, num_layers: int = 24,
                 patch_size: int = 16, mlp_ratio: float = 4.0,
                 num_register_tokens: int = 4, rope_theta: float = 100.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.rope_theta = rope_theta

        self.embeddings = MlxDINOv3PatchEmbed(dim, patch_size, num_register_tokens)
        self.layers = [MlxDINOv3TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(dim)

    def _build_rope_2d(self, h: int, w: int) -> tuple:
        """Build 2D RoPE cos/sin matching HF DINOv3ViTRopePositionEmbedding.

        Returns cos/sin for PATCH tokens only (prefix tokens get no RoPE).
        Shape: (1, h*w, head_dim) — full head_dim, tiled 2x.
        """
        head_dim = self.dim // self.num_heads

        # inv_freq: 1/theta^(arange(0, 1, 4/head_dim)) — matches HF exactly
        inv_freq = 1.0 / (self.rope_theta ** mx.arange(0, 1, 4 / head_dim, dtype=mx.float32))
        # inv_freq shape: (head_dim/4,)

        # Patch center coords normalized to [-1, +1] — matches HF get_patches_center_coordinates
        coords_h = (mx.arange(h, dtype=mx.float32) + 0.5) / h
        coords_w = (mx.arange(w, dtype=mx.float32) + 0.5) / w
        grid_h, grid_w = mx.meshgrid(coords_h, coords_w, indexing='ij')
        coords = mx.stack([grid_h.reshape(-1), grid_w.reshape(-1)], axis=-1)  # (h*w, 2)
        coords = 2.0 * coords - 1.0  # shift to [-1, +1]

        # angles: 2π * coord * inv_freq, then flatten and tile 2x
        angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]  # (h*w, 2, head_dim/4)
        angles = angles.reshape(h * w, -1)  # (h*w, head_dim/2)
        # Tile to full head_dim (matching PT's angles.tile(2))
        angles = mx.concatenate([angles, angles], axis=-1)  # (h*w, head_dim)

        cos = mx.cos(angles)[None, :, :]  # (1, h*w, head_dim)
        sin = mx.sin(angles)[None, :, :]
        return cos, sin

    def __call__(self, images: list) -> mx.array:
        x = self._preprocess(images)
        B, H, W, _ = x.shape

        h = self.embeddings(x)

        nH, nW = H // self.patch_size, W // self.patch_size
        rope_cos, rope_sin = self._build_rope_2d(nH, nW)

        # 1 CLS + num_register_tokens registers = prefix tokens (no RoPE)
        num_prefix = 1 + self.num_register_tokens

        for layer in self.layers:
            h = layer(h, rope_cos, rope_sin, num_prefix_tokens=num_prefix)

        # Match PT reference: pure layer_norm without learned params
        # (PT's extract_features uses F.layer_norm, not the model's learned norm)
        h = mx.fast.layer_norm(h, weight=None, bias=None, eps=1e-5)
        return h

    def _preprocess(self, images: list) -> mx.array:
        """Resize, normalize, convert to MLX array."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                size = max(img.size)
                target = ((size + self.patch_size - 1) // self.patch_size) * self.patch_size
                img = img.resize((target, target), Image.LANCZOS)
                arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
            else:
                arr = np.array(img, dtype=np.float32)
            arr = (arr - mean) / std
            processed.append(arr)

        return mx.array(np.stack(processed))


def load_dinov3_from_hf(
    model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    image_size: int = 512,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> MlxDINOv3FeatureExtractor:
    """Load DINOv3 weights from HuggingFace into MLX model."""
    from huggingface_hub import hf_hub_download
    import json

    hub_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    config_path = hf_hub_download(model_name, "config.json", **hub_kwargs)
    weight_path = hf_hub_download(model_name, "model.safetensors", **hub_kwargs)

    with open(config_path) as f:
        config = json.load(f)

    dim = config.get('hidden_size', 1024)
    num_heads = config.get('num_attention_heads', 16)
    num_layers = config.get('num_hidden_layers', 24)
    patch_size = config.get('patch_size', 16)
    mlp_ratio = config.get('mlp_ratio', 4.0)
    num_register = config.get('num_register_tokens', 4)

    model = MlxDINOv3FeatureExtractor(
        dim=dim, num_heads=num_heads, num_layers=num_layers,
        patch_size=patch_size, mlp_ratio=mlp_ratio,
        num_register_tokens=num_register,
    )

    weights = mx.load(weight_path)
    remapped = _remap_dinov3_weights(weights, num_layers, dim)
    model.load_weights(list(remapped.items()))

    return model


def _remap_dinov3_weights(weights: dict, num_layers: int, dim: int) -> dict:
    """
    Map HuggingFace DINOv3 weight keys to our model structure.

    HF format:
        layer.N.attention.q_proj.weight, layer.N.attention.k_proj.weight (no bias),
        layer.N.attention.v_proj.weight, layer.N.attention.o_proj.weight/bias,
        layer.N.mlp.up_proj.weight/bias, layer.N.mlp.down_proj.weight/bias,
        layer.N.norm1/norm2.weight/bias, layer.N.layer_scale1/2.lambda1

    Our format:
        layers.N.attn.qkv.weight/bias, layers.N.attn.proj.weight/bias,
        layers.N.mlp.fc1.weight/bias, layers.N.mlp.fc2.weight/bias,
        layers.N.norm1/norm2.weight/bias, layers.N.layer_scale1/2
    """
    remapped = {}

    # Embeddings
    if 'embeddings.patch_embeddings.weight' in weights:
        # HF: (dim, 3, P, P) -> our: (dim, P, P, 3)
        remapped['embeddings.weight'] = weights['embeddings.patch_embeddings.weight'].transpose(0, 2, 3, 1)
    if 'embeddings.patch_embeddings.bias' in weights:
        remapped['embeddings.bias'] = weights['embeddings.patch_embeddings.bias']
    if 'embeddings.cls_token' in weights:
        remapped['embeddings.cls_token'] = weights['embeddings.cls_token']
    if 'embeddings.register_tokens' in weights:
        remapped['embeddings.register_tokens'] = weights['embeddings.register_tokens']

    # Final norm
    if 'norm.weight' in weights:
        remapped['norm.weight'] = weights['norm.weight']
    if 'norm.bias' in weights:
        remapped['norm.bias'] = weights['norm.bias']

    # Transformer layers — separate Q/K/V projections (matching HF layout)
    for i in range(num_layers):
        prefix_hf = f'layer.{i}'
        prefix_mlx = f'layers.{i}'

        # Q, K, V projections — keep separate (not fused)
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            for suffix in ['weight', 'bias']:
                key = f'{prefix_hf}.attention.{proj}.{suffix}'
                if key in weights:
                    remapped[f'{prefix_mlx}.attn.{proj}.{suffix}'] = weights[key]

        # Output projection
        for suffix in ['weight', 'bias']:
            key = f'{prefix_hf}.attention.o_proj.{suffix}'
            if key in weights:
                remapped[f'{prefix_mlx}.attn.o_proj.{suffix}'] = weights[key]

        # Norms
        for norm in ['norm1', 'norm2']:
            for suffix in ['weight', 'bias']:
                key = f'{prefix_hf}.{norm}.{suffix}'
                if key in weights:
                    remapped[f'{prefix_mlx}.{norm}.{suffix}'] = weights[key]

        # MLP: up_proj → fc1, down_proj → fc2
        for suffix in ['weight', 'bias']:
            key = f'{prefix_hf}.mlp.up_proj.{suffix}'
            if key in weights:
                remapped[f'{prefix_mlx}.mlp.fc1.{suffix}'] = weights[key]
            key = f'{prefix_hf}.mlp.down_proj.{suffix}'
            if key in weights:
                remapped[f'{prefix_mlx}.mlp.fc2.{suffix}'] = weights[key]

        # Layer scaling
        key1 = f'{prefix_hf}.layer_scale1.lambda1'
        if key1 in weights:
            remapped[f'{prefix_mlx}.layer_scale1'] = weights[key1]
        key2 = f'{prefix_hf}.layer_scale2.lambda1'
        if key2 in weights:
            remapped[f'{prefix_mlx}.layer_scale2'] = weights[key2]

    return remapped
