"""
MLX backend for Trellis2 on Apple Silicon.

All models run in pure MLX — no PyTorch except at the mesh-extraction boundary.
Weight loading uses mx.load() for zero-copy safetensors reads.
"""
import logging
import mlx.core as mx


def setup_logging(level=logging.INFO):
    """Enable MLX backend logging. Call before pipeline.run()."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("mlx_backend").setLevel(level)

__all__ = [
    'load_safetensors',
    'remap_flow_model_weights',
    'remap_vae_decoder_weights',
]


def load_safetensors(path: str, dtype=None) -> dict:
    """Load safetensors weights via MLX.

    Keeps native dtype (bf16/fp16) by default. The RoPE interleaving fix
    ensures numerical correctness regardless of weight precision.
    Pass dtype=mx.float32 to force float32 if needed for debugging.
    """
    weights = mx.load(path)
    if dtype is not None:
        weights = {k: v.astype(dtype) if v.dtype in (mx.bfloat16, mx.float16) else v
                   for k, v in weights.items()}
    return weights


def remap_flow_model_weights(weights: dict) -> dict:
    """
    Remap safetensors keys to MLX module paths for flow models.

    MLX uses plain list indices for model.blocks (no `.layers.`):
      blocks.0.xxx stays as blocks.0.xxx

    But nn.Sequential-like containers use `.layers.`:
      blocks.0.mlp.mlp.0.weight → blocks.0.mlp.mlp.layers.0.weight
      adaLN_modulation.1.weight → adaLN_modulation.layers.1.weight
      t_embedder.mlp.0.weight   → t_embedder.mlp.layers.0.weight
    """
    remapped = {}
    for k, v in weights.items():
        new_k = k
        # blocks.N.xxx stays as-is (MLX uses integer indices for lists)
        # Only remap Sequential containers within blocks

        # Sequential .N. → .layers.N. for specific containers
        new_k = _remap_sequential(new_k)

        remapped[new_k] = v
    return remapped


def remap_vae_decoder_weights(weights: dict) -> dict:
    """
    Remap safetensors keys to MLX module paths for VAE decoders.

    VAE decoder has nested lists: blocks[i][j].xxx
    MLX uses plain integer indices: blocks.i.j.xxx (no .layers.)
    """
    remapped = {}
    for k, v in weights.items():
        new_k = k

        # MlxSparseLinear wraps nn.Linear as self.linear
        # from_latent.weight → from_latent.linear.weight
        # output_layer.weight → output_layer.linear.weight
        # Also handle to_subdiv in upsample blocks
        for linear_name in ['from_latent', 'output_layer', 'to_subdiv', 'skip_connection']:
            if new_k.startswith(f'{linear_name}.'):
                new_k = new_k.replace(f'{linear_name}.', f'{linear_name}.linear.', 1)
                break
            # Handle nested: blocks.X.Y.to_subdiv.weight
            import re
            pattern = rf'(blocks\.\d+\.\d+\.{linear_name})\.'
            new_k_try = re.sub(pattern, rf'\1.linear.', new_k, count=1)
            if new_k_try != new_k:
                new_k = new_k_try
                break

        # Sequential .N. → .layers.N. for MLP-like containers
        new_k = _remap_sequential(new_k)

        remapped[new_k] = v
    return remapped


def _remap_sequential(key: str) -> str:
    """
    Convert PyTorch nn.Sequential index notation to MLX.
    e.g. 'foo.0.weight' → 'foo.layers.0.weight' when '0' is a digit
    within known sequential containers.
    """
    import re
    # Match patterns like: prefix.DIGIT.suffix where prefix ends with
    # a known sequential container name
    sequential_containers = [
        'mlp.mlp', 'adaLN_modulation', 't_embedder.mlp', 'mlp'
    ]
    for container in sequential_containers:
        # Pattern: container.DIGIT.rest
        pattern = re.escape(container) + r'\.(\d+)\.'
        replacement = container + r'.layers.\1.'
        key = re.sub(pattern, replacement, key)
        # Also handle terminal case: container.DIGIT (no trailing dot)
        pattern_end = re.escape(container) + r'\.(\d+)$'
        replacement_end = container + r'.layers.\1'
        key = re.sub(pattern_end, replacement_end, key)
    return key
