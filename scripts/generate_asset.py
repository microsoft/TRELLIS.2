#!/usr/bin/env python3
"""Generate reproducible full-resolution and PBR TRELLIS.2 assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# MPS fallback must be configured before torch or any Metal extension imports.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trellis2.model_revisions import (  # noqa: E402
    DINOV3_REPO,
    DINOV3_REVISION,
    RMBG_REPO,
    RMBG_REVISION,
    SOURCE_REVISIONS,
    TRELLIS_IMAGE_LARGE_REPO,
    TRELLIS_IMAGE_LARGE_REVISION,
    TRELLIS_REPO,
    TRELLIS_REVISION,
)
from trellis2.mesh_integrity import measure_glb  # noqa: E402
from trellis2.backends import fp32_decode_thresholds_enabled  # noqa: E402

SAFETY_FACE_TARGET = 200_000
WATCHDOG_SIGNATURES = (
    "non-zero size",
    "BVH needs at least 8 triangles",
    "kIOGPUCommandBufferCallbackErrorImpactingInteractivity",
    "empty mesh",
)
OOM_SIGNATURES = (
    "out of memory",
    "mps backend out of memory",
    "failed to allocate",
    "allocation failed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_revision() -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _package_versions() -> dict[str, Optional[str]]:
    names = (
        "torch",
        "torchvision",
        "transformers",
        "huggingface-hub",
        "mlx",
        "flex-gemm",
        "mtldiffrast",
        "cumesh",
        "trimesh",
        "xatlas",
        "fast-simplification",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parse_target(value: str) -> Optional[int]:
    if value.lower() in {"none", "full", "off", "0"}:
        return None
    target = int(value)
    if target < 8:
        raise argparse.ArgumentTypeError("decimation target must be at least 8 faces or 'none'")
    return target


def _watchdog_message(error: BaseException) -> str:
    detail = str(error)
    if isinstance(error, MemoryError) or any(
        signature in detail.lower() for signature in OOM_SIGNATURES
    ):
        return (
            "MPS ran out of memory. Close GPU-heavy apps, use one TRELLIS process, "
            "and retry with --pipeline-type 512. Do not disable the MPS high-watermark "
            f"guard. Original error: {detail}"
        )
    if any(signature.lower() in detail.lower() for signature in WATCHDOG_SIGNATURES):
        return (
            "The decoder produced an empty or watchdog-corrupted mesh. Close GPU-heavy apps, "
            "retry pipeline 512, or run with TRELLIS_DISABLE_METAL=1 for the slower fallback. "
            f"Original error: {detail}"
        )
    return detail


def _raw_trimesh(mesh):
    import numpy as np
    import trimesh

    vertices = mesh.vertices.detach().cpu().numpy().astype(np.float32, copy=True)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int64, copy=True)

    # Match o_voxel.postprocess's asset-space conversion exactly.
    converted = vertices.copy()
    converted[:, 1] = vertices[:, 2]
    converted[:, 2] = -vertices[:, 1]
    raw = trimesh.Trimesh(vertices=converted, faces=faces, process=False)
    # trimesh omits NORMAL from glTF when a process=False mesh has not yet
    # materialized its cached vertex normals. Accessing the property computes
    # them without merging vertices or changing the full-resolution topology.
    _ = raw.vertex_normals
    return raw


def _export_raw(mesh, path: Path) -> dict[str, Any]:
    raw = _raw_trimesh(mesh)
    raw.export(path)
    return {
        "vertices": int(len(raw.vertices)),
        "triangles": int(len(raw.faces)),
        "bounds_min": [float(value) for value in raw.bounds[0]],
        "bounds_max": [float(value) for value in raw.bounds[1]],
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _metal_baker_available() -> tuple[bool, str]:
    try:
        from trellis2.backends import probe_metal_backends
        from o_voxel import postprocess

        probes = probe_metal_backends()
        failed_probes = {
            name: result for name, result in probes.items() if not result.get("ok")
        }
        if failed_probes:
            return False, json.dumps(failed_probes, sort_keys=True)
        available = bool(
            getattr(postprocess, "_HAS_DR", False)
            and getattr(postprocess, "_HAS_MESH", False)
            and getattr(postprocess, "_BACKEND", None) == "metal"
        )
        if available:
            return True, ""
        errors = getattr(postprocess, "_BACKEND_ERRORS", {})
        return False, json.dumps(errors, sort_keys=True) if errors else "Metal backend is incomplete"
    except (ImportError, RuntimeError, OSError) as exc:
        return False, str(exc)


def _export_pbr(
    mesh,
    path: Path,
    *,
    baker: str,
    target: Optional[int],
    texture_size: int,
    remesh: bool,
    remesh_band: float,
    remesh_project: float,
    remesh_project_max_dist: float,
    remesh_project_min_agreement: float,
    technical_safety_target: bool,
):
    vertices = mesh.vertices.cpu()
    faces = mesh.faces.cpu()
    pre_simplified = False

    # Keep the legacy pre-BVH reduction only for a non-remesh technical fallback.
    # Remesh accepts arbitrary input size and must receive the full mesh;
    # raw_full.glb is exported separately and remains untouched.
    if (
        not remesh
        and technical_safety_target
        and target is not None
        and int(faces.shape[0]) > target
    ):
        import fast_simplification
        import torch

        reduction = 1.0 - (target / int(faces.shape[0]))
        simplified_vertices, simplified_faces = fast_simplification.simplify(
            vertices.numpy(),
            faces.numpy(),
            target_reduction=reduction,
        )
        vertices = torch.from_numpy(simplified_vertices).to(dtype=mesh.vertices.dtype)
        faces = torch.from_numpy(simplified_faces).to(dtype=mesh.faces.dtype)
        pre_simplified = True

    if baker == "metal":
        available, reason = _metal_baker_available()
        if not available:
            raise RuntimeError(f"Metal baker unavailable: {reason}")
        from o_voxel import postprocess

        exporter = postprocess.to_glb
    elif baker == "kdtree":
        from o_voxel import postprocess_cpu

        exporter = postprocess_cpu.to_glb
    else:
        raise ValueError(f"Unknown baker: {baker}")

    result = exporter(
        vertices=vertices,
        faces=faces,
        attr_volume=mesh.attrs.cpu(),
        coords=mesh.coords.cpu(),
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
        remesh_project_max_dist=remesh_project_max_dist,
        remesh_project_min_agreement=remesh_project_min_agreement,
        verbose=True,
    )
    result.export(path)
    return result, pre_simplified


def _attempt_schedule(preferred: str, requested_target: Optional[int], raw_faces: int):
    preferred_order = ["metal", "kdtree"] if preferred in {"auto", "metal"} else ["kdtree"]
    targets = [requested_target]
    if requested_target is None and raw_faces > SAFETY_FACE_TARGET:
        targets.append(SAFETY_FACE_TARGET)
    seen = set()
    for target in targets:
        for baker in preferred_order:
            key = (baker, target)
            if key not in seen:
                seen.add(key)
                yield baker, target


def _run_pbr_attempts(
    mesh,
    candidate_path: Path,
    *,
    preferred_baker: str,
    requested_target: Optional[int],
    texture_size: int,
    remesh: bool = True,
    remesh_band: float = 1.0,
    remesh_project: float = 0.7,
    remesh_project_max_dist: float = 1.5,
    remesh_project_min_agreement: float = 0.5,
    export_fn=None,
    on_attempt=None,
):
    """Run the fixed Metal/KDTree/full/safety fallback order."""
    export_fn = export_fn or _export_pbr
    raw_faces = int(mesh.faces.shape[0])
    attempts = []
    for baker, target in _attempt_schedule(preferred_baker, requested_target, raw_faces):
        attempt = {
            "baker": baker,
            "target_faces": target,
            "remeshed": False,
            "remesh_band": remesh_band,
            "remesh_project": remesh_project,
            "remesh_project_max_dist": remesh_project_max_dist,
            "remesh_project_min_agreement": remesh_project_min_agreement,
            "technical_safety_target": (
                requested_target is None
                and target == SAFETY_FACE_TARGET
                and raw_faces > SAFETY_FACE_TARGET
            ),
        }
        attempt_started = time.perf_counter()
        try:
            exported, pre_simplified = export_fn(
                mesh,
                candidate_path,
                baker=baker,
                target=target,
                texture_size=texture_size,
                remesh=remesh,
                remesh_band=remesh_band,
                remesh_project=remesh_project,
                remesh_project_max_dist=remesh_project_max_dist,
                remesh_project_min_agreement=remesh_project_min_agreement,
                technical_safety_target=attempt["technical_safety_target"],
            )
            attempt["status"] = "ok"
            attempt["pre_simplified_before_bvh"] = pre_simplified
            attempt["remeshed"] = bool(baker == "metal" and remesh)
        except Exception as exc:  # each failure is recorded before the prescribed fallback
            exported = None
            attempt["status"] = "failed"
            attempt["error_type"] = type(exc).__name__
            attempt["error"] = str(exc)
            if candidate_path.exists():
                candidate_path.unlink()
        attempt["duration_seconds"] = round(time.perf_counter() - attempt_started, 3)
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)
        if exported is not None:
            return exported, attempt, attempts

    errors = "; ".join(
        f"{attempt['baker']}@{attempt['target_faces']}: {attempt.get('error', 'failed')}"
        for attempt in attempts
    )
    raise RuntimeError(f"All PBR export attempts failed: {errors}")


def _record_integrity_metadata(
    metadata: dict[str, Any], raw_path: Path, candidate_path: Path
) -> None:
    candidate_integrity = measure_glb(candidate_path)
    raw_integrity = measure_glb(raw_path)
    metadata["candidate_pbr"]["integrity"] = candidate_integrity
    metadata["raw_full"]["integrity"] = raw_integrity
    metadata["candidate_pbr"]["promotable"] = bool(
        metadata["candidate_pbr"]["remeshed"]
        and candidate_integrity["boundary_edges"] <= 8
        and candidate_integrity["winding_consistent"]
    )


def _candidate_stats(path: Path, exported) -> dict[str, Any]:
    import numpy as np

    vertices = np.asarray(exported.vertices)
    faces = np.asarray(exported.faces)
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "bounds_min": [float(value) for value in vertices.min(axis=0)],
        "bounds_max": [float(value) for value in vertices.max(axis=0)],
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _load_pipeline(args):
    import torch

    if args.backend == "mlx-experimental":
        from huggingface_hub import snapshot_download
        from mlx_backend.pipeline import create_mlx_pipeline

        snapshot = snapshot_download(
            TRELLIS_REPO,
            revision=TRELLIS_REVISION,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )
        return create_mlx_pipeline(
            snapshot,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
        ), "mlx-experimental"

    if args.backend == "mps":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("The mps backend requires an Apple Silicon Mac")
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is not available in this Python environment")
    elif args.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("auto selected CUDA, but PyTorch CUDA is unavailable")
    else:
        raise RuntimeError(f"Unsupported resolved backend: {args.backend}")

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        TRELLIS_REPO,
        revision=TRELLIS_REVISION,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if args.backend == "mps":
        pipeline.to(torch.device("mps"))
        return pipeline, "mps"
    if args.backend == "cuda":
        pipeline.cuda()
        return pipeline, "cuda"
    raise AssertionError("resolved backend validation fell through")


def _resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"

    # Keep the official Linux/CUDA route intact without adding a new public
    # CLI choice: --backend auto resolves to CUDA on supported non-Mac hosts.
    import torch

    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("auto found neither Apple MPS nor NVIDIA CUDA")


def _parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "mps", "mlx-experimental"), default="auto")
    parser.add_argument("--baker", choices=("auto", "metal", "kdtree"), default="auto")
    parser.add_argument("--pipeline-type", choices=("512", "1024", "1024_cascade"), default="512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--texture-size", type=int, choices=(512, 1024, 2048), default=1024)
    parser.add_argument("--background", choices=("auto", "keep"), default="auto")
    parser.add_argument("--pbr-decimation-target", type=_parse_target, default=None, metavar="none|N")
    remesh_group = parser.add_mutually_exclusive_group()
    remesh_group.add_argument("--remesh", dest="remesh", action="store_true")
    remesh_group.add_argument("--no-remesh", dest="remesh", action="store_false")
    parser.set_defaults(remesh=True)
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project", type=float, default=0.7)
    parser.add_argument("--remesh-project-max-dist", type=float, default=1.5)
    parser.add_argument("--remesh-project-min-agreement", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    requested_backend = args.backend
    args.image = args.image.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = str(args.cache_dir.expanduser().resolve()) if args.cache_dir else None

    if not args.image.is_file():
        raise SystemExit(f"Input image not found: {args.image}")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    args.backend = _resolve_backend(args.backend)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_full.glb"
    candidate_path = args.output_dir / "candidate_pbr.glb"
    metadata_path = args.output_dir / "meta.json"
    existing = [path for path in (raw_path, candidate_path, metadata_path) if path.exists()]
    if existing and not args.force:
        raise SystemExit("Refusing to overwrite existing outputs: " + ", ".join(map(str, existing)))
    if args.force:
        for path in existing:
            path.unlink()

    from PIL import Image

    with Image.open(args.image) as opened:
        image = opened.copy()
        input_size = list(opened.size)
        input_mode = opened.mode

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "input": {
            "path": str(args.image),
            "sha256": _sha256(args.image),
            "size": input_size,
            "mode": input_mode,
        },
        "configuration": {
            "backend": requested_backend,
            "resolved_backend": args.backend,
            "preferred_baker": args.baker,
            "pipeline_type": args.pipeline_type,
            "seed": args.seed,
            "steps": args.steps,
            "texture_size": args.texture_size,
            "background": args.background,
            "pbr_decimation_target": args.pbr_decimation_target,
            "remesh": args.remesh,
            "remesh_band": args.remesh_band,
            "remesh_project": args.remesh_project,
            "remesh_project_max_dist": args.remesh_project_max_dist,
            "remesh_project_min_agreement": args.remesh_project_min_agreement,
            "fp32_decode_thresholds": fp32_decode_thresholds_enabled(),
            "offline": args.offline,
            "cache_dir": args.cache_dir,
        },
        "revisions": {
            "code": _git_revision(),
            TRELLIS_REPO: TRELLIS_REVISION,
            TRELLIS_IMAGE_LARGE_REPO: TRELLIS_IMAGE_LARGE_REVISION,
            DINOV3_REPO: DINOV3_REVISION,
            RMBG_REPO: RMBG_REVISION,
        },
        "source_revisions": SOURCE_REVISIONS,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": _package_versions(),
        "timings_seconds": {},
        "pbr_attempts": [],
    }
    _write_json(metadata_path, metadata)

    started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        pipeline, resolved_backend = _load_pipeline(args)
        metadata["configuration"]["resolved_backend"] = resolved_backend
        try:
            from trellis2.backends import backend_report

            metadata["backend_capabilities"] = backend_report()
        except (ImportError, RuntimeError, OSError) as exc:
            metadata["backend_capabilities"] = {"probe_error": str(exc)}
        metadata["timings_seconds"]["pipeline_load"] = round(time.perf_counter() - load_started, 3)
        _write_json(metadata_path, metadata)

        sampler_overrides = {"steps": args.steps} if args.steps is not None else {}
        generation_started = time.perf_counter()
        try:
            outputs = pipeline.run(
                image,
                seed=args.seed,
                pipeline_type=args.pipeline_type,
                preprocess_image=args.background == "auto",
                sparse_structure_sampler_params=sampler_overrides,
                shape_slat_sampler_params=sampler_overrides,
                tex_slat_sampler_params=sampler_overrides,
            )
        except (IndexError, AssertionError, RuntimeError, MemoryError) as exc:
            raise RuntimeError(_watchdog_message(exc)) from exc
        metadata["timings_seconds"]["generation"] = round(time.perf_counter() - generation_started, 3)

        mesh = outputs[0] if isinstance(outputs, list) else outputs
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] < 8:
            raise RuntimeError(_watchdog_message(RuntimeError("empty mesh")))

        raw_started = time.perf_counter()
        metadata["raw_full"] = _export_raw(mesh, raw_path)
        metadata["timings_seconds"]["raw_export"] = round(time.perf_counter() - raw_started, 3)
        _write_json(metadata_path, metadata)

        pbr_started = time.perf_counter()
        raw_faces = int(mesh.faces.shape[0])

        def record_attempt(attempt):
            metadata["pbr_attempts"].append(attempt)
            _write_json(metadata_path, metadata)

        exported, chosen, _ = _run_pbr_attempts(
            mesh,
            candidate_path,
            preferred_baker=args.baker,
            requested_target=args.pbr_decimation_target,
            texture_size=args.texture_size,
            remesh=args.remesh,
            remesh_band=args.remesh_band,
            remesh_project=args.remesh_project,
            remesh_project_max_dist=args.remesh_project_max_dist,
            remesh_project_min_agreement=args.remesh_project_min_agreement,
            on_attempt=record_attempt,
        )

        metadata["candidate_pbr"] = _candidate_stats(candidate_path, exported)
        metadata["candidate_pbr"].update(chosen)
        metadata["candidate_pbr"]["decimated"] = bool(
            chosen["target_faces"] is not None and chosen["target_faces"] < raw_faces
        )
        metadata["candidate_pbr"]["technical_decimation"] = bool(
            metadata["candidate_pbr"]["decimated"]
            and chosen["technical_safety_target"]
        )
        _record_integrity_metadata(metadata, raw_path, candidate_path)
        metadata["timings_seconds"]["pbr_export"] = round(time.perf_counter() - pbr_started, 3)
        from trellis2.gltf_validation import validate_output_pair

        validation_started = time.perf_counter()
        metadata["validation"] = validate_output_pair(raw_path, candidate_path)
        metadata["timings_seconds"]["validation"] = round(
            time.perf_counter() - validation_started, 3
        )
        metadata["timings_seconds"]["total"] = round(time.perf_counter() - started, 3)
        metadata["peak_rss_bytes"] = _peak_rss_bytes()
        metadata["status"] = "ok"
        _write_json(metadata_path, metadata)

        print(f"raw_full.glb: {metadata['raw_full']['triangles']:,} triangles")
        print(
            "candidate_pbr.glb: "
            f"{metadata['candidate_pbr']['triangles']:,} triangles via {chosen['baker']}"
        )
        print(f"metadata: {metadata_path}")
        return 0
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        metadata["timings_seconds"]["total"] = round(time.perf_counter() - started, 3)
        metadata["peak_rss_bytes"] = _peak_rss_bytes()
        _write_json(metadata_path, metadata)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
