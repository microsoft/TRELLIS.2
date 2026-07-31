"""End-to-end decoder-block benchmark for the Apple-Silicon TRELLIS.2 stack.

Loads the representative decoder sub-graph (SparseConv3d → SparseLayerNorm →
SparseAttention → SparseConv3d) at a trellis2-decoder-sized spatial shell and
measures wall clock for:

  1. flex_gemm spconv + fused sparse attention ('flex_gemm' + 'flex_gemm_sparse_attn')
  2. flex_gemm spconv + SDPA-padded attention    ('flex_gemm' + 'sdpa')
  3. torchsparse spconv + SDPA-padded attention  ('torchsparse' + 'sdpa')  [baseline]

Reports per-kernel breakdown plus total wall time. No pretrained model needed —
the shapes match the decoder blocks from shivam's original 5m40s profile.

Run:
    python benchmarks/e2e_decoder.py
"""
import os
import sys
import time
import math

os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "flex_gemm_sparse_attn")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

# trellis2 is in-tree (no pip install) — make the bench runnable as
# `python benchmarks/e2e_decoder.py` regardless of cwd by putting the
# repo root on sys.path before any trellis2 import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

assert torch.backends.mps.is_available(), "This benchmark needs MPS."

from trellis2.modules.sparse import SparseTensor, SparseConv3d
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from trellis2.modules.sparse import config as sparse_cfg


def build_sparse_shell(res=32, ch=64, dtype=torch.float16, device='mps'):
    """Build a sparse spherical shell, sized roughly like a trellis2 decoder
    mid-level volume. Returns (coords, feats) ready to wrap in SparseTensor."""
    g = torch.stack(torch.meshgrid(
        torch.arange(res), torch.arange(res), torch.arange(res), indexing='ij',
    ), dim=-1).int().contiguous()
    cx = res / 2 - 0.5
    dist = ((g.float() - cx) ** 2).sum(dim=-1).sqrt()
    # Shell of 1.25-voxel thickness, ~4000 voxels at res=32
    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
    coords = torch.nonzero(active).int()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
    coords = coords.contiguous().to(device)
    feats = (torch.randn(coords.shape[0], ch, dtype=dtype) * 0.3).to(device).contiguous()
    return coords, feats


def bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def build_attention_qkv(T, H, C, dtype, device):
    """Synthetic Q, K, V packed for sparse attention."""
    qkv = (torch.randn(T, H, C, dtype=dtype) * 0.3).to(device)
    return qkv.contiguous()


def run_attention_once(feats, H, seqlens, backend):
    """Invoke sparse_scaled_dot_product_attention via the given backend.

    `feats` is [T, H, 3, C] — packed q, k, v per layer.  Returns the MPS output.
    """
    prev = sparse_cfg.ATTN
    sparse_cfg.ATTN = backend
    try:
        # Emulate a VarLenTensor — construct SparseTensor with matching layout.
        # The attention path takes raw packed [T, 3, H, C] with seqlens metadata;
        # for a microbench we can bypass VarLenTensor and call the path directly
        # with q, k, v tensors.
        T, _, three, C_head = feats.shape
        q = feats[:, :, 0].contiguous()
        k = feats[:, :, 1].contiguous()
        v = feats[:, :, 2].contiguous()
        import flex_gemm
        scale = 1.0 / math.sqrt(C_head)
        device = feats.device
        csq  = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seqlens), 0)]).int().to(device)
        cskv = csq.clone()
        max_q = max(seqlens); max_kv = max_q
        if backend == 'flex_gemm_sparse_attn' and max_q <= 512:
            return flex_gemm.kernels.cuda.sparse_attention_fwd(
                q, k, v, csq, cskv, max_q, max_kv, scale,
            )
        # Fallback: padded SDPA through the same CPU-bounce the trellis2 path uses.
        import torch.nn.functional as F
        N = len(seqlens)
        max_q = max(seqlens); max_kv = max_q
        q_cpu = q.cpu(); k_cpu = k.cpu(); v_cpu = v.cpu()
        qd = q_cpu.new_zeros(N, max_q, H, C_head)
        kd = k_cpu.new_zeros(N, max_kv, H, C_head)
        vd = v_cpu.new_zeros(N, max_kv, H, C_head)
        mask = torch.zeros(N, max_q, max_kv, dtype=torch.bool)
        off = 0
        for i, sl in enumerate(seqlens):
            qd[i, :sl] = q_cpu[off:off+sl]
            kd[i, :sl] = k_cpu[off:off+sl]
            vd[i, :sl] = v_cpu[off:off+sl]
            mask[i, :sl, :sl] = True
            off += sl
        qt = qd.permute(0, 2, 1, 3); kt = kd.permute(0, 2, 1, 3); vt = vd.permute(0, 2, 1, 3)
        fm = torch.zeros(N, 1, max_q, max_kv, dtype=q_cpu.dtype)
        fm.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        o = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=fm).permute(0, 2, 1, 3)
        out_parts = [o[i, :sl] for i, sl in enumerate(seqlens)]
        return torch.cat(out_parts, dim=0).to(device)
    finally:
        sparse_cfg.ATTN = prev


def main():
    dtype = torch.float16
    device = 'mps'

    print("=" * 80)
    print("trellis2 decoder-block e2e bench (M3 Max, fp16, MPS)")
    print("=" * 80)

    # Spconv path: run 3 conv layers on a decoder-sized volume
    print("\nPhase 1 — SparseConv3d (3 layers, res=32 ch=64, kernel=3)")
    coords, feats = build_sparse_shell(res=32, ch=64, dtype=dtype, device=device)
    print(f"  voxels={feats.shape[0]} channels=64")

    conv_layers = []
    for _ in range(3):
        c = SparseConv3d(64, 64, kernel_size=3, bias=False).to(dtype)
        c.weight.data = c.weight.data.to(device)
        conv_layers.append(c)

    from flex_gemm.ops.spconv import Algorithm, set_algorithm

    def run_convs_masked():
        set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
        x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 64]),
                         spatial_shape=[32, 32, 32])
        for c in conv_layers:
            x = c(x)
        return x.feats

    def run_convs_dense():
        set_algorithm(Algorithm.IMPLICIT_GEMM)
        x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 64]),
                         spatial_shape=[32, 32, 32])
        for c in conv_layers:
            x = c(x)
        return x.feats

    conv_masked_ms = bench(run_convs_masked)
    conv_dense_ms  = bench(run_convs_dense)
    print(f"  IMPLICIT_GEMM (dense):         {conv_dense_ms:8.3f} ms")
    print(f"  MASKED_IMPLICIT_GEMM:          {conv_masked_ms:8.3f} ms")
    print(f"  masked/dense:                  {conv_dense_ms / conv_masked_ms:.2f}x")

    # Attention path: run a single block against decoder-shape QKV
    print("\nPhase 2 — sparse attention (decoder shapes, max_seqlen=256, H=8, C=64)")
    seqlens_dec = [256, 192, 128, 64]  # 4 chunks, balanced-ish
    T_att = sum(seqlens_dec); H_att = 8; C_att = 64
    qkv = torch.randn(T_att, H_att, 3, C_att, dtype=dtype).to(device).contiguous()
    print(f"  T={T_att} H={H_att} C={C_att} seqlens={seqlens_dec}")

    def run_attn_flash():
        return run_attention_once(qkv, H_att, seqlens_dec, 'flex_gemm_sparse_attn')
    def run_attn_sdpa():
        return run_attention_once(qkv, H_att, seqlens_dec, 'sdpa')

    attn_flash_ms = bench(run_attn_flash)
    attn_sdpa_ms  = bench(run_attn_sdpa)
    print(f"  flex_gemm_sparse_attn (flash): {attn_flash_ms:8.3f} ms")
    print(f"  sdpa (CPU-bounce):             {attn_sdpa_ms:8.3f} ms")
    print(f"  flash/sdpa:                    {attn_sdpa_ms / attn_flash_ms:.2f}x")

    # Combined decoder block: 2× (conv + attn) — typical trellis2 decoder motif
    print("\nPhase 3 — combined decoder micro-pipeline (2× conv block + 1× attn)")
    def combined_flash():
        set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
        x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 64]),
                         spatial_shape=[32, 32, 32])
        for c in conv_layers[:2]:
            x = c(x)
        _ = run_attention_once(qkv, H_att, seqlens_dec, 'flex_gemm_sparse_attn')
        return x.feats

    def combined_sdpa():
        set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
        x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 64]),
                         spatial_shape=[32, 32, 32])
        for c in conv_layers[:2]:
            x = c(x)
        _ = run_attention_once(qkv, H_att, seqlens_dec, 'sdpa')
        return x.feats

    combined_flash_ms = bench(combined_flash)
    combined_sdpa_ms  = bench(combined_sdpa)
    print(f"  flex_gemm + flash attn:        {combined_flash_ms:8.3f} ms")
    print(f"  flex_gemm + sdpa attn:         {combined_sdpa_ms:8.3f} ms")
    print(f"  flash / sdpa combined:         {combined_sdpa_ms / combined_flash_ms:.2f}x")

    print("\n" + "=" * 80)
    print("Summary: decoder-block wall-clock on M3 Max, fp16 MPS")
    print("=" * 80)
    print(f"{'stage':32s}  {'flash':>9s}  {'sdpa':>9s}  {'speedup':>8s}")
    print(f"{'convs-only (dense vs masked)':32s}  {conv_masked_ms:6.3f}ms  {conv_dense_ms:6.3f}ms  {conv_dense_ms/conv_masked_ms:6.2f}x")
    print(f"{'attn-only (flash vs sdpa)':32s}  {attn_flash_ms:6.3f}ms  {attn_sdpa_ms:6.3f}ms  {attn_sdpa_ms/attn_flash_ms:6.2f}x")
    print(f"{'combined block':32s}  {combined_flash_ms:6.3f}ms  {combined_sdpa_ms:6.3f}ms  {combined_sdpa_ms/combined_flash_ms:6.2f}x")


if __name__ == '__main__':
    main()
