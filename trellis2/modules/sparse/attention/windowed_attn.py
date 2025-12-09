from typing import *
import torch
import math
from .. import SparseTensor
from .. import config


__all__ = [
    'sparse_windowed_scaled_dot_product_self_attention',
    'sparse_windowed_scaled_dot_product_cross_attention',
]


def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (torch.Tensor): Sequence lengths.
        (dict): Attn func args.
    """
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = [i + j for i, j in zip(tensor.spatial_shape, shift_window)]
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    mask = seq_lens != 0
    seq_lens = seq_lens[mask]
    
    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        attn_func_args = {
            'attn_bias': xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
        }
    elif config.ATTN == 'flash_attn':
        attn_func_args = {
            'cu_seqlens': torch.cat([torch.tensor([0], device=tensor.device), torch.cumsum(seq_lens, dim=0)], dim=0).int(),
            'max_seqlen': torch.max(seq_lens)
        }

    return fwd_indices, bwd_indices, seq_lens, attn_func_args
    

def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        
    Returns:
        (SparseTensor): [N, *, H, C] sparse tensor containing the output features.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'windowed_attention_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, attn_func_args))
    else:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = serialization_spatial_cache
    
    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]

    if config.DEBUG:
        start = 0
        qkv_coords = qkv.coords[fwd_indices]
        for i in range(len(seq_lens)):
            seq_coords = qkv_coords[start:start+seq_lens[i]]
            assert (seq_coords[:, 1:].max(dim=0).values - seq_coords[:, 1:].min(dim=0).values < window_size).all(), \
                    f"SparseWindowedScaledDotProductSelfAttention: window size exceeded"
            start += seq_lens[i]

    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        q, k, v = qkv_feats.unbind(dim=1)                                               # [M, H, C]
        q = q.unsqueeze(0)                                                              # [1, M, H, C]
        k = k.unsqueeze(0)                                                              # [1, M, H, C]
        v = v.unsqueeze(0)                                                              # [1, M, H, C]
        out = xops.memory_efficient_attention(q, k, v, **attn_func_args)[0]             # [M, H, C]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)  # [M, H, C]

    out = out[bwd_indices]      # [T, H, C]

    if config.DEBUG:
        qkv_coords = qkv_coords[bwd_indices]
        assert torch.equal(qkv_coords, qkv.coords), "SparseWindowedScaledDotProductSelfAttention: coordinate mismatch"

    return qkv.replace(out)


def sparse_windowed_scaled_dot_product_cross_attention(
    q: SparseTensor,
    kv: SparseTensor,
    q_window_size: int,
    kv_window_size: int,
    q_shift_window: Tuple[int, int, int] = (0, 0, 0),
    kv_shift_window: Tuple[int, int, int] = (0, 0, 0),
) -> SparseTensor:
    """
    Apply windowed scaled dot product cross attention to two sparse tensors.

    Args:
        q (SparseTensor): [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor): [N, *, 2, H, C] sparse tensor containing Ks and Vs.
        q_window_size (int): The window size to use for Qs.
        kv_window_size (int): The window size to use for Ks and Vs.
        q_shift_window (Tuple[int, int, int]): The shift of serialized coordinates for Qs.
        kv_shift_window (Tuple[int, int, int]): The shift of serialized coordinates for Ks and Vs.
        
    Returns:
        (SparseTensor): [N, *, H, C] sparse tensor containing the output features.
    """
    assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
    assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"

    q_serialization_spatial_cache_name = f'windowed_attention_{q_window_size}_{q_shift_window}'
    q_serialization_spatial_cache = q.get_spatial_cache(q_serialization_spatial_cache_name)
    if q_serialization_spatial_cache is None:
        q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args = calc_window_partition(q, q_window_size, q_shift_window)
        q.register_spatial_cache(q_serialization_spatial_cache_name, (q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args))
    else:
        q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args = q_serialization_spatial_cache
    kv_serialization_spatial_cache_name = f'windowed_attention_{kv_window_size}_{kv_shift_window}'
    kv_serialization_spatial_cache = kv.get_spatial_cache(kv_serialization_spatial_cache_name)
    if kv_serialization_spatial_cache is None:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args = calc_window_partition(kv, kv_window_size, kv_shift_window)
        kv.register_spatial_cache(kv_serialization_spatial_cache_name, (kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args))
    else:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args = kv_serialization_spatial_cache

    assert len(q_seq_lens) == len(kv_seq_lens), "Number of sequences in q and kv must match"

    q_feats = q.feats[q_fwd_indices]      # [M, H, C]
    kv_feats = kv.feats[kv_fwd_indices]    # [M, 2, H, C]

    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        k, v = kv_feats.unbind(dim=1)                                                   # [M, H, C]
        q = q.unsqueeze(0)                                                              # [1, M, H, C]
        k = k.unsqueeze(0)                                                              # [1, M, H, C]
        v = v.unsqueeze(0)                                                              # [1, M, H, C]
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seq_lens, kv_seq_lens)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)[0]               # [M, H, C]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats,
            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],
            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],
        )  # [M, H, C]

    out = out[q_bwd_indices]      # [T, H, C]

    return q.replace(out)
