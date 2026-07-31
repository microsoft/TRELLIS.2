from typing import *
import torch
import torch.nn.functional as F
import math
from .. import SparseTensor
from .. import config


__all__ = [
    'sparse_windowed_scaled_dot_product_self_attention',
    'sparse_windowed_scaled_dot_product_cross_attention',
]


def _sdpa_varlen_qkvpacked(qkv_feats: torch.Tensor, attn_func_args: dict) -> torch.Tensor:
    """sdpa for variable-length qkv-packed input (self-attention within windows)."""
    q, k, v = qkv_feats.unbind(dim=1)  # each [M, H, C]
    cu_seqlens = attn_func_args['cu_seqlens']
    seq_lens = attn_func_args['seq_lens']
    N = len(seq_lens)
    max_len = attn_func_args['max_seqlen'].item() if isinstance(attn_func_args['max_seqlen'], torch.Tensor) else attn_func_args['max_seqlen']
    H, C = q.shape[-2], q.shape[-1]

    # Pad into dense batch [N, max_len, H, C]
    q_dense = q.new_zeros(N, max_len, H, C)
    k_dense = k.new_zeros(N, max_len, H, C)
    v_dense = v.new_zeros(N, max_len, H, C)
    mask = torch.zeros(N, max_len, dtype=torch.bool, device=q.device)
    for i in range(N):
        sl = seq_lens[i].item() if isinstance(seq_lens[i], torch.Tensor) else seq_lens[i]
        start = cu_seqlens[i].item()
        q_dense[i, :sl] = q[start:start + sl]
        k_dense[i, :sl] = k[start:start + sl]
        v_dense[i, :sl] = v[start:start + sl]
        mask[i, :sl] = True

    # [N, H, L, C]
    q_dense = q_dense.permute(0, 2, 1, 3)
    k_dense = k_dense.permute(0, 2, 1, 3)
    v_dense = v_dense.permute(0, 2, 1, 3)

    # Build float mask for MPS compatibility
    sdpa_mask = mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, L]
    float_mask = torch.zeros(N, 1, max_len, max_len, dtype=q_dense.dtype, device=q.device)
    float_mask.masked_fill_(~(mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(3)), float('-inf'))

    out_dense = F.scaled_dot_product_attention(q_dense, k_dense, v_dense, attn_mask=float_mask)
    out_dense = out_dense.permute(0, 2, 1, 3)  # [N, L, H, C]

    # Unpad
    parts = []
    for i in range(N):
        sl = seq_lens[i].item() if isinstance(seq_lens[i], torch.Tensor) else seq_lens[i]
        parts.append(out_dense[i, :sl])
    return torch.cat(parts, dim=0)


def _sdpa_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 q_args: dict, kv_args: dict) -> torch.Tensor:
    """sdpa for variable-length cross-attention within windows."""
    q_cu = q_args['cu_seqlens']
    kv_cu = kv_args['cu_seqlens']
    q_seq_lens = q_args['seq_lens']
    kv_seq_lens = kv_args['seq_lens']
    N = len(q_seq_lens)
    max_q = q_args['max_seqlen'].item() if isinstance(q_args['max_seqlen'], torch.Tensor) else q_args['max_seqlen']
    max_kv = kv_args['max_seqlen'].item() if isinstance(kv_args['max_seqlen'], torch.Tensor) else kv_args['max_seqlen']
    H, C_q = q.shape[-2], q.shape[-1]
    C_v = v.shape[-1]

    q_dense = q.new_zeros(N, max_q, H, C_q)
    k_dense = k.new_zeros(N, max_kv, H, C_q)
    v_dense = v.new_zeros(N, max_kv, H, C_v)
    q_mask = torch.zeros(N, max_q, dtype=torch.bool, device=q.device)
    kv_mask = torch.zeros(N, max_kv, dtype=torch.bool, device=q.device)
    for i in range(N):
        ql = q_seq_lens[i].item() if isinstance(q_seq_lens[i], torch.Tensor) else q_seq_lens[i]
        kvl = kv_seq_lens[i].item() if isinstance(kv_seq_lens[i], torch.Tensor) else kv_seq_lens[i]
        qs = q_cu[i].item()
        kvs = kv_cu[i].item()
        q_dense[i, :ql] = q[qs:qs + ql]
        k_dense[i, :kvl] = k[kvs:kvs + kvl]
        v_dense[i, :kvl] = v[kvs:kvs + kvl]
        q_mask[i, :ql] = True
        kv_mask[i, :kvl] = True

    q_dense = q_dense.permute(0, 2, 1, 3)
    k_dense = k_dense.permute(0, 2, 1, 3)
    v_dense = v_dense.permute(0, 2, 1, 3)

    # q_mask: (N, max_q), kv_mask: (N, max_kv) -> cross mask: (N, 1, max_q, max_kv)
    cross_mask = q_mask.unsqueeze(2) & kv_mask.unsqueeze(1)  # (N, max_q, max_kv)
    float_mask = torch.zeros(N, 1, max_q, max_kv, dtype=q_dense.dtype, device=q.device)
    float_mask.masked_fill_(~cross_mask.unsqueeze(1), float('-inf'))

    out_dense = F.scaled_dot_product_attention(q_dense, k_dense, v_dense, attn_mask=float_mask)
    out_dense = out_dense.permute(0, 2, 1, 3)

    parts = []
    for i in range(N):
        ql = q_seq_lens[i].item() if isinstance(q_seq_lens[i], torch.Tensor) else q_seq_lens[i]
        parts.append(out_dense[i, :ql])
    return torch.cat(parts, dim=0)


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
    elif config.ATTN == 'sdpa':
        attn_func_args = {
            'cu_seqlens': torch.cat([torch.tensor([0], device=tensor.device), torch.cumsum(seq_lens, dim=0)], dim=0).int(),
            'max_seqlen': torch.max(seq_lens),
            'seq_lens': seq_lens,
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
    elif config.ATTN == 'sdpa':
        out = _sdpa_varlen_qkvpacked(qkv_feats, attn_func_args)                         # [M, H, C]

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
        q_feats_u = q_feats.unsqueeze(0)                                                # [1, M, H, C]
        k = k.unsqueeze(0)                                                              # [1, M, H, C]
        v = v.unsqueeze(0)                                                              # [1, M, H, C]
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seq_lens, kv_seq_lens)
        out = xops.memory_efficient_attention(q_feats_u, k, v, attn_bias=mask)[0]       # [M, H, C]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats,
            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],
            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],
        )  # [M, H, C]
    elif config.ATTN == 'sdpa':
        k, v = kv_feats.unbind(dim=1)
        out = _sdpa_varlen(q_feats, k, v, q_attn_func_args, kv_attn_func_args)          # [M, H, C]

    out = out[q_bwd_indices]      # [T, H, C]

    return q.replace(out)
