"""End-to-end integration smoke test for SPARSE_CONV_BACKEND=flex_gemm on MPS.

Mirrors what trellis2's sparse decoder does: builds a SparseTensor on MPS,
runs a SparseConv3d through the flex_gemm backend, and feeds the result
through a LayerNorm — the exact path that originally crashed with
"Passed CPU tensor to MPS op" before mtlgemm's device-routing fix.

Exercises Algorithm.IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM, and the
production default Algorithm.MASKED_IMPLICIT_GEMM_SPLITK. All should produce
numerically equivalent output and pass the LayerNorm hand-off without
crashing.

Note: this script intentionally uses torch.nn.functional.layer_norm rather
than trellis2's SparseLayerNorm wrapper, because that wrapper currently calls
torch.zeros_like which lacks an MPS kernel in some PyTorch builds. That is a
PyTorch issue, separate from the mtlgemm fix being verified here.
"""

# This is an intentionally standalone, verbose MPS integration program rather
# than a pytest module. Avoid executing its device setup during pytest
# collection; CI and macOS acceptance invoke it directly.
if __name__ != "__main__":
    import pytest

    pytest.skip("run test_flex_gemm_integration.py directly", allow_module_level=True)

import os
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import torch

assert torch.backends.mps.is_available(), "This test needs MPS"

from trellis2.modules.sparse import SparseTensor, SparseConv3d
from trellis2.modules.sparse.conv import config as conv_config
from flex_gemm.ops.spconv import Algorithm

device = "mps"
dtype = torch.float16

# Build a small sparse voxel shell (mimics trellis2 decoder input scale)
res = 16
ch = 32
coords = torch.stack(torch.meshgrid(
    torch.arange(res), torch.arange(res), torch.arange(res), indexing="ij",
), dim=-1).int().contiguous()
dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
coords = torch.nonzero(active).int()
coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
coords = coords.contiguous().to(device)
feats = torch.randn(coords.shape[0], ch, dtype=dtype).to(device).contiguous()

print(f"Sparse tensor: {feats.shape[0]} voxels, {ch} channels, dtype={dtype}, device={device}")

x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, ch]), spatial_shape=[res, res, res])
print(f"Built SparseTensor: feats device={x.feats.device}, dtype={x.feats.dtype}")

# Build the conv module on CPU then move to MPS to dodge LayerNorm-init MPS limits.
conv = SparseConv3d(ch, ch, kernel_size=3, bias=True).to(dtype)
conv.weight.data = conv.weight.data.to(device)
if conv.bias is not None:
    conv.bias.data = conv.bias.data.to(device)
print(f"Conv weight device: {conv.weight.device}, dtype: {conv.weight.dtype}")

def _run_with_algo(algo, label):
    # conv_flex_gemm sets flex_gemm's global algorithm from this config on
    # every forward call, so changing flex_gemm directly would be overwritten.
    conv_config.FLEX_GEMM_ALGO = algo
    # Neighbor-cache layouts are algorithm-specific. Use a fresh sparse tensor
    # so this parity test cannot feed one algorithm another algorithm's cache.
    run_x = SparseTensor(
        feats=feats,
        coords=coords,
        shape=torch.Size([1, ch]),
        spatial_shape=[res, res, res],
    )
    out = conv(run_x)
    assert out.feats.device.type == "mps", f"FAIL [{label}]: SparseConv3d on {out.feats.device}, expected mps"
    assert out.feats.dtype == dtype, f"FAIL [{label}]: SparseConv3d dtype {out.feats.dtype}, expected {dtype}"
    # LayerNorm hand-off — the original crash site
    w = torch.ones(ch, dtype=dtype).to(device)
    b = torch.zeros(ch, dtype=dtype).to(device)
    z = torch.nn.functional.layer_norm(out.feats, (ch,), w, b)
    assert z.device.type == "mps", f"FAIL [{label}]: LayerNorm on {z.device}"
    total = z.sum().item()
    print(f"  {label:30s} feats={tuple(out.feats.shape)} sum={total:.4f}")
    return out.feats.detach().cpu().float()

print()
y_dense  = _run_with_algo(Algorithm.IMPLICIT_GEMM, "IMPLICIT_GEMM (dense)")
y_masked = _run_with_algo(Algorithm.MASKED_IMPLICIT_GEMM, "MASKED_IMPLICIT_GEMM")
y_splitk = _run_with_algo(Algorithm.MASKED_IMPLICIT_GEMM_SPLITK, "MASKED_IMPLICIT_GEMM_SPLITK")

# Numerical parity between the two algorithms — same inputs, equivalent output.
diff = (y_dense - y_masked).abs().max().item()
splitk_diff = (y_dense - y_splitk).abs().max().item()
parity_tol = 2e-2  # fp16 — masked reduces in a different order
assert diff <= parity_tol, f"FAIL: dense vs masked diff {diff:.4e} > tol {parity_tol:.4e}"
assert splitk_diff <= parity_tol, f"FAIL: dense vs split-k diff {splitk_diff:.4e} > tol {parity_tol:.4e}"
print(f"  parity dense vs masked: max_diff={diff:.4e} (tol={parity_tol})")
print(f"  parity dense vs split-k: max_diff={splitk_diff:.4e} (tol={parity_tol})")

print()
print("PASS — trellis2-apple SparseConv3d + LayerNorm runs end-to-end on MPS via flex_gemm")
print("       Dense, masked, and production split-k algorithms produce equivalent output.")
