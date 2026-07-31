from typing import *
import math
import platform
import sys

CONV = 'flex_gemm'
DEBUG = False
ATTN = 'flash_attn'

def __detect_defaults():
    """Auto-detect best backends for current platform."""
    global CONV, ATTN
    if platform.system() == 'Darwin':
        if __flex_gemm_works_on_mps():
            CONV = 'flex_gemm'
            # Sparse convolution and sparse attention are separate Metal
            # kernels. A working convolution install does not prove that the
            # attention entry point is ABI-compatible with the active torch
            # build, so select it only after its own numerical probe.
            ATTN = (
                'flex_gemm_sparse_attn'
                if probe_flex_gemm_sparse_attention_on_mps()
                else 'sdpa'
            )
        else:
            CONV = 'pytorch'
            ATTN = 'sdpa'
    elif not __has_cuda():
        CONV = 'pytorch'
        ATTN = 'sdpa'


def __flex_gemm_works_on_mps():
    """Probe dense, masked, and production split-k flex_gemm kernels on MPS.

    If the install pre-dates the device-routing fix (or a real masked kernel),
    one of these returns a CPU tensor or raises. Fall back to pure PyTorch
    rather than crashing inside the model on the first LayerNorm. Build tensors
    on CPU and move to MPS because some PyTorch builds lack int/fp16 creation
    kernels on MPS.
    """
    try:
        import os
        if os.environ.get('TRELLIS_DISABLE_METAL', '0') == '1':
            return False
        import torch
        if not torch.backends.mps.is_available():
            return False
        import flex_gemm
        from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm

        # Exercise both algorithms — masked carries its own cache/dispatch path
        # distinct from dense. A stale install may have one working and the
        # other broken (e.g. the pre-round-2 aliased-to-dense fallback).
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32).to('mps')
        feats = torch.zeros((1, 4), dtype=torch.float16).to('mps')
        weight = torch.zeros((4, 1, 1, 1, 4), dtype=torch.float16).to('mps')
        shape = torch.Size([1, 4, 1, 1, 1])

        for algo in (
            Algorithm.IMPLICIT_GEMM,
            Algorithm.MASKED_IMPLICIT_GEMM,
            Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
        ):
            set_algorithm(algo)
            out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight)
            torch.mps.synchronize()
            if out.device.type != 'mps' or not bool(torch.isfinite(out).all().item()):
                return False
        return True
    except Exception:
        return False


def probe_flex_gemm_sparse_attention_on_mps() -> bool:
    """Exercise the real Metal sparse-attention kernel and compare it to SDPA.

    This intentionally uses the production head dimension (64) while keeping
    the sequence tiny. Returning ``False`` is a controlled capability result:
    callers fall back to PyTorch SDPA without disabling working Metal sparse
    convolution kernels.
    """
    try:
        import os
        if os.environ.get('TRELLIS_DISABLE_METAL', '0') == '1':
            return False

        import torch
        import torch.nn.functional as F
        if not torch.backends.mps.is_available():
            return False

        import flex_gemm

        tokens, heads, head_dim = 16, 2, 64
        generator = torch.Generator(device='cpu').manual_seed(42)
        q = torch.randn(tokens, heads, head_dim, dtype=torch.float16, generator=generator).to('mps').contiguous()
        k = torch.randn(tokens, heads, head_dim, dtype=torch.float16, generator=generator).to('mps').contiguous()
        v = torch.randn(tokens, heads, head_dim, dtype=torch.float16, generator=generator).to('mps').contiguous()
        cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32).to('mps')

        out = flex_gemm.kernels.cuda.sparse_attention_fwd(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            tokens,
            tokens,
            1.0 / math.sqrt(head_dim),
        )
        reference = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        ).squeeze(0).transpose(0, 1)
        torch.mps.synchronize()

        return (
            out.device.type == 'mps'
            and out.shape == reference.shape
            and bool(torch.isfinite(out).all().item())
            and bool(torch.allclose(out, reference, rtol=2e-2, atol=2e-2))
        )
    except Exception:
        return False

def __has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def __from_env():
    import os

    global CONV
    global DEBUG
    global ATTN

    __detect_defaults()

    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm', 'pytorch']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in [
        'xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'flex_gemm_sparse_attn',
    ]:
        ATTN = env_sparse_attn_backend

    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")


__from_env()


def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm', 'pytorch']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'flex_gemm_sparse_attn']):
    global ATTN
    ATTN = backend
