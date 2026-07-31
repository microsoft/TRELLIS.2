import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS backend default")
def test_dense_attention_defaults_to_sdpa_on_macos():
    env = os.environ.copy()
    env.pop("ATTN_BACKEND", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from trellis2.modules.attention import config; print('BACKEND=' + config.BACKEND)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "BACKEND=sdpa" in completed.stdout


def test_disable_metal_selects_controlled_pytorch_fallback():
    env = os.environ.copy()
    env.update(
        {
            "TRELLIS_DISABLE_METAL": "1",
            "PYTHONPATH": str(ROOT),
            "SPARSE_CONV_BACKEND": "pytorch",
            "SPARSE_ATTN_BACKEND": "sdpa",
        }
    )
    code = """
import json
from trellis2.modules.sparse import config
from trellis2.backends import backend_report
print('REPORT=' + json.dumps({
    'conv': config.CONV,
    'attention': config.ATTN,
    'backend': backend_report(),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    line = next(line for line in completed.stdout.splitlines() if line.startswith("REPORT="))
    report = json.loads(line.removeprefix("REPORT="))
    assert report["conv"] == "pytorch"
    assert report["attention"] == "sdpa"
    assert report["backend"]["metal_disabled"] is True
    assert report["backend"]["rasterizer"] is None
    assert report["backend"]["mesh"] is None


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal probe requires macOS")
def test_metal_sparse_attention_has_its_own_numerical_probe():
    # Pipeline imports performed by other test modules may already have made a
    # controlled fallback choice. Exercise auto-detection in a fresh process,
    # which also mirrors a real CLI invocation.
    code = r"""
import json
import torch
if not torch.backends.mps.is_available():
    raise SystemExit(77)
from trellis2.modules.sparse import config
from trellis2.backends import probe_metal_backends
print('REPORT=' + json.dumps({
    'probe': config.probe_flex_gemm_sparse_attention_on_mps(),
    'conv': config.CONV,
    'attention': config.ATTN,
    'metal': probe_metal_backends(),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip("MPS is unavailable")
    assert completed.returncode == 0, completed.stderr
    line = next(line for line in completed.stdout.splitlines() if line.startswith("REPORT="))
    report = json.loads(line.removeprefix("REPORT="))
    assert report["probe"]
    assert report["conv"] == "flex_gemm"
    assert report["attention"] == "flex_gemm_sparse_attn"
    assert report["metal"]["rasterizer"]["ok"], report["metal"]["rasterizer"]
    assert report["metal"]["mesh_bvh"]["ok"], report["metal"]["mesh_bvh"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal parity requires macOS")
def test_pure_pytorch_sparse_conv_matches_flex_gemm_on_mps():
    code = r"""
import torch
from trellis2.modules.sparse import SparseTensor, config
from trellis2.modules.sparse.conv.conv import SparseConv3d

if not torch.backends.mps.is_available():
    raise SystemExit(77)
coords = torch.tensor(
    [[0, x, y, z] for x in range(3) for y in range(3) for z in range(3)],
    dtype=torch.int32,
    device='mps',
)
features = torch.randn(27, 4, dtype=torch.float16, device='mps')
sparse = SparseTensor(
    feats=features,
    coords=coords,
    shape=torch.Size([1, 4]),
    spatial_shape=[3, 3, 3],
)
config.set_conv_backend('flex_gemm')
metal = SparseConv3d(4, 5, 3, bias=True).half().to('mps')
metal_out = metal(sparse).feats

config.set_conv_backend('pytorch')
fallback = SparseConv3d(4, 5, 3, bias=True).half().to('mps')
fallback.weight.data.copy_(metal.weight.data)
fallback.bias.data.copy_(metal.bias.data)
fallback_out = fallback(sparse).feats
torch.mps.synchronize()
print('PARITY=' + str(float((metal_out - fallback_out).abs().max().item())))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip("MPS is unavailable")
    assert completed.returncode == 0, completed.stderr
    parity_line = next(line for line in completed.stdout.splitlines() if line.startswith("PARITY="))
    assert float(parity_line.removeprefix("PARITY=")) <= 2e-2
