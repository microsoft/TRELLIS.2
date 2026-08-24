"""
Shared helper to detect whether flash_attn is actually usable on the current
GPU/CUDA build. Import succeeding is not sufficient: flash_attn can be
installed but fail at runtime on unsupported architectures or CUDA/torch
version mismatches. The result is cached so the (relatively expensive) CUDA
probe only runs once per process even though both the dense and sparse
attention config modules need it.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def flash_attn_usable() -> bool:
    try:
        import torch
        import flash_attn
        if not torch.cuda.is_available():
            return False
        q = torch.zeros(1, 1, 1, 8, dtype=torch.float16, device='cuda')
        flash_attn.flash_attn_func(q, q, q)
        return True
    except Exception:
        return False
