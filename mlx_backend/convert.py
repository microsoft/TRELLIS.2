"""
Conversion utilities between PyTorch and MLX tensors.
Used ONLY at pipeline boundaries (mesh extraction input).
"""
import numpy as np
import mlx.core as mx


def torch_to_mlx(t) -> mx.array:
    """Convert a PyTorch tensor to MLX array."""
    import torch
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().cpu().numpy()).astype(mx.bfloat16)
    return mx.array(t.cpu().numpy())


def mlx_to_torch(a: mx.array):
    """Convert an MLX array to PyTorch tensor."""
    import torch
    arr = np.array(a)
    return torch.from_numpy(arr)


def mlx_to_numpy(a: mx.array) -> np.ndarray:
    """Convert an MLX array to numpy."""
    return np.array(a)
