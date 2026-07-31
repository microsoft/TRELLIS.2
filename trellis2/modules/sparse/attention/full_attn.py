from typing import *
import torch
from .. import VarLenTensor
from .. import config


__all__ = [
    'sparse_scaled_dot_product_attention',
]


@overload
def sparse_scaled_dot_product_attention(qkv: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (VarLenTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, kv: Union[VarLenTensor, torch.Tensor]) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (VarLenTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, C] dense tensor containing Qs.
        kv (VarLenTensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: VarLenTensor, v: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: torch.Tensor, v: torch.Tensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: VarLenTensor, v: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, VarLenTensor), f"qkv must be a VarLenTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, VarLenTensor) and isinstance(kv, (VarLenTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, VarLenTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, VarLenTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, VarLenTensor) and isinstance(k, (VarLenTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, VarLenTensor) and isinstance(v, VarLenTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, VarLenTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    elif config.ATTN == 'flash_attn_3':
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            cu_seqlens_kv = cu_seqlens_q.clone()
            max_q_seqlen = max_kv_seqlen = max(q_seqlen)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        elif num_all_args == 3:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)
    elif config.ATTN == 'flex_gemm_sparse_attn':
        # Fused variable-length sparse attention (mtlgemm Metal kernel).
        # Dispatch is unconditional, matching the CUDA backends above
        # (xformers / flash_attn / flash_attn_3) which never fork on
        # max_seqlen. The underlying kernel is flash-attention-v2 with
        # simdgroup_matrix_multiply_accumulate for Q@K^T and P@V, and
        # parallelized online-softmax row reductions via simd_shuffle_xor.
        #
        # Optional safety-valve: FLEX_GEMM_ATTN_MAX_SEQLEN=N falls back to
        # the SDPA-padded path when max(q_seqlen, kv_seqlen) > N. Useful on
        # PyTorch builds where the Accelerate-SDPA-CPU-bounce is actually
        # faster at the specific shapes the user hits (rare; measured
        # crossover sits beyond 768 on fp32 and likely higher on fp16).
        import os as _os
        _cap_env = _os.environ.get('FLEX_GEMM_ATTN_MAX_SEQLEN', '0')
        try:
            _cap = int(_cap_env)
        except ValueError:
            _cap = 0
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        _use_fused = (_cap <= 0) or (max(max(q_seqlen), max(kv_seqlen)) <= _cap)
        if _use_fused:
            import flex_gemm, math
            scale = 1.0 / math.sqrt(q.shape[-1])
            csq = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), 0)]).int().to(device)
            cskv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), 0)]).int().to(device)
            out = flex_gemm.kernels.cuda.sparse_attention_fwd(
                q.contiguous(), k.contiguous(), v.contiguous(),
                csq, cskv, max(q_seqlen), max(kv_seqlen), scale,
            )
        else:
            # FLEX_GEMM_ATTN_MAX_SEQLEN safety-valve active — fall back to
            # SDPA-padded. Mirror the SDPA preamble (q/k/v already unbound).
            import torch.nn.functional as F_attn
            N_b = len(q_seqlen)
            max_q = max(q_seqlen); max_kv = max(kv_seqlen)
            H_b = q.shape[-2]; C_q_b = q.shape[-1]; C_v_b = v.shape[-1]
            q_dense = q.new_zeros(N_b, max_q, H_b, C_q_b)
            k_dense = k.new_zeros(N_b, max_kv, H_b, C_q_b)
            v_dense = v.new_zeros(N_b, max_kv, H_b, C_v_b)
            attn_mask = torch.zeros(N_b, max_q, max_kv, dtype=torch.bool, device=device)
            q_off = 0; kv_off = 0
            for i in range(N_b):
                ql = q_seqlen[i]; kvl = kv_seqlen[i]
                q_dense[i, :ql] = q[q_off:q_off + ql]
                k_dense[i, :kvl] = k[kv_off:kv_off + kvl]
                v_dense[i, :kvl] = v[kv_off:kv_off + kvl]
                attn_mask[i, :ql, :kvl] = True
                q_off += ql; kv_off += kvl
            q_dense = q_dense.permute(0, 2, 1, 3)
            k_dense = k_dense.permute(0, 2, 1, 3)
            v_dense = v_dense.permute(0, 2, 1, 3)
            float_mask = torch.zeros(N_b, 1, max_q, max_kv, dtype=q_dense.dtype, device=device)
            float_mask.masked_fill_(~attn_mask.unsqueeze(1), float('-inf'))
            out_dense = F_attn.scaled_dot_product_attention(q_dense, k_dense, v_dense, attn_mask=float_mask)
            out_dense = out_dense.permute(0, 2, 1, 3)
            out_parts = [out_dense[i, :q_seqlen[i]] for i in range(N_b)]
            out = torch.cat(out_parts, dim=0)
    elif config.ATTN == 'sdpa':
        import torch.nn.functional as F_attn
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        # Pad variable-length sequences into dense batch [N, max_len, H, C]
        N = len(q_seqlen)
        max_q = max(q_seqlen)
        max_kv = max(kv_seqlen)
        H = q.shape[-2]
        C_q = q.shape[-1]
        C_v = v.shape[-1]
        # Build dense tensors
        q_dense = q.new_zeros(N, max_q, H, C_q)
        k_dense = k.new_zeros(N, max_kv, H, C_q)
        v_dense = v.new_zeros(N, max_kv, H, C_v)
        # Build attention mask
        attn_mask = torch.zeros(N, max_q, max_kv, dtype=torch.bool, device=device)
        q_offset = 0
        kv_offset = 0
        for i in range(N):
            ql = q_seqlen[i]
            kvl = kv_seqlen[i]
            q_dense[i, :ql] = q[q_offset:q_offset + ql]
            k_dense[i, :kvl] = k[kv_offset:kv_offset + kvl]
            v_dense[i, :kvl] = v[kv_offset:kv_offset + kvl]
            attn_mask[i, :ql, :kvl] = True
            q_offset += ql
            kv_offset += kvl
        # sdpa expects [N, H, L, C], mask broadcastable to [N, H, Lq, Lkv]
        q_dense = q_dense.permute(0, 2, 1, 3)  # [N, H, Lq, C]
        k_dense = k_dense.permute(0, 2, 1, 3)  # [N, H, Lkv, C]
        v_dense = v_dense.permute(0, 2, 1, 3)  # [N, H, Lkv, C_v]
        # Expand mask for heads: [N, 1, Lq, Lkv]
        sdpa_mask = attn_mask.unsqueeze(1)
        # Use float mask for MPS compatibility (bool masks not supported)
        float_mask = torch.zeros_like(sdpa_mask, dtype=q_dense.dtype)
        float_mask.masked_fill_(~sdpa_mask, float('-inf'))
        out_dense = F_attn.scaled_dot_product_attention(q_dense, k_dense, v_dense, attn_mask=float_mask)
        # out_dense: [N, H, Lq, C_v] -> unpad back to packed
        out_dense = out_dense.permute(0, 2, 1, 3)  # [N, Lq, H, C_v]
        out_parts = []
        for i in range(N):
            out_parts.append(out_dense[i, :q_seqlen[i]])
        out = torch.cat(out_parts, dim=0)
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")
    
    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
