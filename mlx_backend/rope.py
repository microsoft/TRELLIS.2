"""
3D Rotary Position Embeddings for MLX.
Uses a custom Metal kernel for fused application.
"""
import mlx.core as mx
import math


def build_rope_freqs(head_dim: int, dim: int = 3,
                     rope_freq: tuple = (1.0, 10000.0)) -> mx.array:
    """
    Build frequency table for RoPE.

    Returns:
        freqs: (freq_dim,) array of frequencies
    """
    freq_dim = head_dim // 2 // dim
    freqs = mx.arange(freq_dim, dtype=mx.float32) / freq_dim
    freqs = rope_freq[0] / (rope_freq[1] ** freqs)
    return freqs


def compute_rope_phases(coords: mx.array, freqs: mx.array, head_dim: int) -> mx.array:
    """
    Compute RoPE cos/sin phases from 3D coordinates.

    Args:
        coords: (N, 3) int coordinates
        freqs: (freq_dim,) frequency table
        head_dim: head dimension

    Returns:
        cos_phases: (N, head_dim//2) cos values
        sin_phases: (N, head_dim//2) sin values
    """
    N = coords.shape[0]
    dim = coords.shape[1]
    freq_dim = freqs.shape[0]
    target = head_dim // 2

    # Vectorized: compute all phases in one matmul-like op
    # coords: (N, 3) float, freqs: (freq_dim,)
    # phases[d] = coords[:, d:d+1] * freqs -> (N, freq_dim) per dim
    coords_f = coords.astype(mx.float32)
    phases_list = []
    for d in range(dim):
        phases_list.append(coords_f[:, d:d+1] * freqs[None, :])
    phases = mx.concatenate(phases_list, axis=-1)  # (N, dim * freq_dim)

    if phases.shape[-1] < target:
        pad_n = target - phases.shape[-1]
        phases = mx.concatenate([phases, mx.zeros((N, pad_n))], axis=-1)

    cos_phases = mx.cos(phases)
    sin_phases = mx.sin(phases)
    return cos_phases, sin_phases


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """
    Apply rotary embeddings using interleaved pairing (matching PyTorch complex multiply).

    PyTorch pairs adjacent elements: (x[0],x[1]), (x[2],x[3]), ...
    via view_as_complex → complex multiply → view_as_real.

    Args:
        x: (..., N, H, D) features — (N, H, D) or (B, N, H, D)
        cos: (N, D//2) cos phases
        sin: (N, D//2) sin phases

    Returns:
        Same shape as x, rotated.
    """
    # Interleaved: pair (x[0],x[1]), (x[2],x[3]), etc.
    # Reshape last dim from D to (D//2, 2)
    orig_shape = x.shape
    x_paired = x.reshape(*orig_shape[:-1], -1, 2)  # (..., D//2, 2)
    x1 = x_paired[..., 0]  # even indices: (..., D//2)
    x2 = x_paired[..., 1]  # odd indices:  (..., D//2)

    if x.ndim == 3:
        # (N, H, D//2) — expand cos/sin to (N, 1, D//2)
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    else:
        # (B, N, H, D//2) — expand cos/sin to (1, N, 1, D//2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

    # Complex multiply: (x1 + i*x2) * (cos + i*sin)
    o1 = x1 * cos - x2 * sin  # real part
    o2 = x1 * sin + x2 * cos  # imag part

    # Interleave back: stack on last dim then flatten
    out = mx.stack([o1, o2], axis=-1)  # (..., D//2, 2)
    return out.reshape(orig_shape)
