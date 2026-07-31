#!/usr/bin/env python3
"""Benchmark the production Metal narrow-band-DC remesh and cleanup chain."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trellis2.mesh_integrity import measure  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, metavar="INPUT.glb")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--band", type=float, default=1.0)
    parser.add_argument("--project", type=float, default=0.0)
    parser.add_argument(
        "--no-guard",
        action="store_true",
        help="Use the remesher's original unguarded project_back path for A/B runs.",
    )
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument(
        "--cleanup-ops",
        default="dedup,small_components,fill_holes,unify",
        help=(
            "Comma-separated cleanup ops applied after remesh, each recorded "
            "as its own stage: dedup, repair_nm, small_components, fill_holes, "
            "unify. Use 'none' to skip cleanup entirely."
        ),
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix for output file names, to keep runs distinct.",
    )
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    return args


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _peak_rss_mb() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return round(peak_bytes / (1024 * 1024), 3)


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh in {path}, got {type(loaded).__name__}")

    exported_vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    if exported_vertices.ndim != 2 or exported_vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (N, 3), got {exported_vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangle faces with shape (M, 3), got {faces.shape}")
    if exported_vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("Input mesh is empty")

    # scripts/generate_asset.py::_raw_trimesh exports (x, y, z) as
    # (x, z, -y). Invert that transform to recover pipeline asset space.
    vertices = np.array(exported_vertices, dtype=np.float32, copy=True)
    vertices[:, 1] = -exported_vertices[:, 2]
    vertices[:, 2] = exported_vertices[:, 1]
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def _as_numpy(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    return (
        vertices.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
    )


def _record_stage(
    report: dict[str, Any],
    name: str,
    seconds: float,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    **details: Any,
) -> None:
    vertices_np, faces_np = _as_numpy(vertices, faces)
    stage = {
        "name": name,
        "seconds": round(seconds, 3),
        "vertices": int(vertices_np.shape[0]),
        "faces": int(faces_np.shape[0]),
        "integrity": measure(vertices_np, faces_np),
    }
    stage.update(details)
    report["stages"].append(stage)


def _export_mesh(path: Path, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    asset_vertices, asset_faces = _as_numpy(vertices, faces)

    # Reapply the raw_full.glb asset-space to glTF-space conversion exactly.
    converted = np.array(asset_vertices, dtype=np.float32, copy=True)
    converted[:, 1] = asset_vertices[:, 2]
    converted[:, 2] = -asset_vertices[:, 1]
    exported = trimesh.Trimesh(
        vertices=converted,
        faces=np.asarray(asset_faces, dtype=np.int64),
        process=False,
    )
    # Match generate_asset.py's raw export without attaching a material.
    _ = exported.vertex_normals
    exported.export(path)


def main() -> int:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_stem = (
        f"{input_path.stem}_remeshed_r{args.resolution}_p{args.project}"
        + (f"_{args.tag}" if args.tag else "")
    )
    output_path = out_dir / f"{output_stem}.glb"
    report_path = out_dir / f"{output_stem}.report.json"
    report: dict[str, Any] = {
        "input": str(input_path),
        "params": {
            "resolution": args.resolution,
            "band": args.band,
            "project": args.project,
            "no_guard": args.no_guard,
            "skip_prefill": args.skip_prefill,
            "cleanup_ops": args.cleanup_ops,
        },
        "stages": [],
        "peak_rss_mb": None,
        "prefill_failed": False,
    }
    current_stage = "load"

    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input mesh not found: {input_path}")

        vertices_np, faces_np = _load_mesh(input_path)
        vertices = torch.from_numpy(vertices_np).to(dtype=torch.float32, device="cpu")
        faces = torch.from_numpy(faces_np).to(dtype=torch.int32, device="cpu")

        current_stage = "backend_import"
        import o_voxel

        postprocess = o_voxel.postprocess
        if not postprocess._HAS_MESH:
            raise RuntimeError(
                "o_voxel.postprocess has no mesh backend: "
                f"{postprocess._BACKEND_ERRORS.get('cumesh', 'unknown error')}"
            )
        mesh_backend = postprocess._MeshBackend
        bvh_backend = postprocess._BVH
        remesh_narrow_band_dc = postprocess._remesh_narrow_band_dc

        current_stage = "prefill"
        stage_started = time.perf_counter()
        if not args.skip_prefill:
            original_vertices, original_faces = vertices, faces
            try:
                prefill_mesh = mesh_backend()
                prefill_mesh.init(vertices, faces)
                prefill_mesh.fill_holes(max_hole_perimeter=3e-2)
                vertices, faces = prefill_mesh.read()
            except Exception as exc:
                vertices, faces = original_vertices, original_faces
                report["prefill_failed"] = True
                print(
                    f"Prefill failed; continuing with the original mesh: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
        stage_seconds = time.perf_counter() - stage_started
        _record_stage(report, "prefill", stage_seconds, vertices, faces)
        _write_json(report_path, report)

        current_stage = "bvh"
        stage_started = time.perf_counter()
        bvh = bvh_backend(vertices, faces)
        stage_seconds = time.perf_counter() - stage_started
        _record_stage(report, "bvh", stage_seconds, vertices, faces)
        _write_json(report_path, report)

        current_stage = "remesh"
        aabb = torch.stack(
            [vertices.min(dim=0).values, vertices.max(dim=0).values]
        )
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        remesh_domain_scale = (
            (args.resolution + 3 * args.band) / args.resolution * scale
        )
        source_vertices, source_faces = vertices, faces
        stage_started = time.perf_counter()
        vertices, faces = remesh_narrow_band_dc(
            source_vertices,
            source_faces,
            center=center,
            scale=remesh_domain_scale,
            resolution=args.resolution,
            band=args.band,
            project_back=args.project if args.no_guard else 0,
            verbose=True,
            bvh=bvh,
        )
        moved_count = int(vertices.shape[0]) if args.no_guard and args.project > 0 else 0
        reverted_count = 0
        if args.project > 0 and not args.no_guard:
            vertices, moved_count, reverted_count = postprocess._guarded_project_back(
                vertices,
                faces,
                source_vertices,
                source_faces,
                bvh,
                strength=args.project,
                voxel_size=remesh_domain_scale / args.resolution,
                verbose=True,
            )
        stage_seconds = time.perf_counter() - stage_started
        _record_stage(
            report,
            "remesh",
            stage_seconds,
            vertices,
            faces,
            moved_count=moved_count,
            reverted_count=reverted_count,
        )
        _write_json(report_path, report)

        cleanup_ops = [op for op in args.cleanup_ops.split(",") if op and op != "none"]
        for op_name in cleanup_ops:
            current_stage = f"cleanup:{op_name}"
            stage_started = time.perf_counter()
            cleanup_mesh = mesh_backend()
            cleanup_mesh.init(vertices, faces)
            if op_name == "dedup":
                cleanup_mesh.remove_duplicate_faces()
            elif op_name == "repair_nm":
                cleanup_mesh.repair_non_manifold_edges()
            elif op_name == "small_components":
                cleanup_mesh.remove_small_connected_components(1e-5)
            elif op_name == "fill_holes":
                cleanup_mesh.fill_holes(max_hole_perimeter=3e-2)
            elif op_name == "unify":
                cleanup_mesh.unify_face_orientations()
            else:
                raise ValueError(f"Unknown cleanup op: {op_name}")
            vertices, faces = cleanup_mesh.read()
            stage_seconds = time.perf_counter() - stage_started
            _record_stage(report, current_stage, stage_seconds, vertices, faces)
            _write_json(report_path, report)

        current_stage = "export"
        _export_mesh(output_path, vertices, faces)

        report["peak_rss_mb"] = _peak_rss_mb()
        _write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        report["peak_rss_mb"] = _peak_rss_mb()
        report["error"] = {
            "stage": current_stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"ERROR during {current_stage}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
