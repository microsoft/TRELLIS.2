#!/usr/bin/env python3
"""Probe the supported MPS, Metal, SDPA, MLX, and fallback paths."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "o-voxel"))


def _metal_compiler() -> dict:
    result = subprocess.run(
        ["xcrun", "--find", "metal"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "available": result.returncode == 0,
        "path": result.stdout.strip() or None,
        "error": result.stderr.strip() or None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-metal", action="store_true")
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    import torch
    import torch.nn.functional as torch_f
    from scipy.spatial import cKDTree

    report = {
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "metal_compiler": _metal_compiler(),
    }
    failures = []

    if not report["mps_available"]:
        failures.append("PyTorch MPS is unavailable")
    else:
        device = torch.device("mps")
        left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        report["mps_matmul_sum"] = float((left @ left.T).sum().cpu())
        query = torch.randn(2, 2, 8, 16, device=device)
        sdpa = torch_f.scaled_dot_product_attention(query, query, query)
        report["sdpa_shape"] = list(sdpa.shape)

    mlx_matrix = mx.arange(16, dtype=mx.float32).reshape(4, 4)
    mlx_value = mx.sum(mlx_matrix @ mlx_matrix.T)
    mx.eval(mlx_value)
    report["mlx_matmul_sum"] = float(mlx_value.item())

    tree = cKDTree(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    distance, index = tree.query([[0.1, 0.1, 0.1]], k=1)
    report["kdtree_probe"] = {"distance": float(distance[0]), "index": int(index[0])}

    try:
        from trellis2.backends import backend_report

        report["backends"] = backend_report()
    except (ImportError, RuntimeError, OSError) as exc:
        report["backends"] = {"probe_error": str(exc)}

    try:
        from trellis2.modules.sparse import config as sparse_config

        report["sparse_backends"] = {
            "convolution": sparse_config.CONV,
            "attention": sparse_config.ATTN,
            "metal_attention_parity": (
                sparse_config.probe_flex_gemm_sparse_attention_on_mps()
            ),
        }
    except (ImportError, RuntimeError, OSError) as exc:
        report["sparse_backends"] = {"probe_error": str(exc)}

    if args.require_metal:
        backends = report.get("backends", {})
        capability_probes = backends.get("capability_probes", {})
        sparse_backends = report.get("sparse_backends", {})
        if not report["metal_compiler"]["available"]:
            failures.append("Xcode Metal compiler is unavailable")
        if backends.get("rasterizer") != "metal":
            failures.append("mtldiffrast Metal rasterizer failed capability probe")
        if backends.get("mesh") != "metal":
            failures.append("mtlmesh/mtlbvh failed capability probe")
        if not backends.get("flex_gemm"):
            failures.append("mtlgemm/flex_gemm failed capability probe")
        if sparse_backends.get("convolution") != "flex_gemm":
            failures.append("flex_gemm sparse-convolution capability probe failed")
        if not capability_probes.get("rasterizer", {}).get("ok"):
            failures.append("mtldiffrast raster capability probe failed")
        if not capability_probes.get("mesh_bvh", {}).get("ok"):
            failures.append("mtlmesh/mtlbvh capability probe failed")

    report["ok"] = not failures
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
