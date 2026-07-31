import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import generate_asset


def _mesh(face_count):
    return SimpleNamespace(faces=np.zeros((face_count, 3), dtype=np.int64))


def test_default_has_no_decimation_limit():
    assert generate_asset._parse_target("none") is None
    assert generate_asset._parse_target("0") is None
    assert generate_asset._parse_target("250000") == 250000


def test_remesh_cli_defaults_and_overrides():
    defaults = generate_asset._parse_args(["input.png", "--output-dir", "output"])
    assert defaults.remesh is True
    assert defaults.remesh_band == 1.0
    assert defaults.remesh_project == 0.7
    assert defaults.remesh_project_max_dist == 1.5
    assert defaults.remesh_project_min_agreement == 0.5

    disabled = generate_asset._parse_args(
        [
            "input.png",
            "--output-dir",
            "output",
            "--no-remesh",
            "--remesh-band",
            "1.5",
            "--remesh-project",
            "0.25",
            "--remesh-project-max-dist",
            "2.25",
            "--remesh-project-min-agreement",
            "0.75",
        ]
    )
    assert disabled.remesh is False
    assert disabled.remesh_band == 1.5
    assert disabled.remesh_project == 0.25
    assert disabled.remesh_project_max_dist == 2.25
    assert disabled.remesh_project_min_agreement == 0.75

    enabled = generate_asset._parse_args(
        ["input.png", "--output-dir", "output", "--remesh"]
    )
    assert enabled.remesh is True


def test_cpu_baker_warns_once_when_remesh_is_requested(monkeypatch):
    from o_voxel import postprocess_cpu

    class StopAfterWarning(Exception):
        pass

    def stop_before_export():
        raise StopAfterWarning

    monkeypatch.setattr(postprocess_cpu, "_get_device", stop_before_export)

    with pytest.warns(RuntimeWarning) as recorded:
        with pytest.raises(StopAfterWarning):
            postprocess_cpu.to_glb(None, None, None, None, {}, None, remesh=True)

    assert len(recorded) == 1
    assert str(recorded[0].message) == (
        "kdtree/CPU path cannot remesh; candidate will not be promotable"
    )


def test_oom_diagnostic_recommends_safe_single_process_retry():
    message = generate_asset._watchdog_message(RuntimeError("MPS backend out of memory"))
    assert "--pipeline-type 512" in message
    assert "one TRELLIS process" in message


def test_auto_backend_selects_mps_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(generate_asset.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(generate_asset.platform, "machine", lambda: "arm64")
    assert generate_asset._resolve_backend("auto") == "mps"


def test_auto_backend_preserves_linux_cuda_route(monkeypatch):
    monkeypatch.setattr(generate_asset.platform, "system", lambda: "Linux")
    monkeypatch.setattr(generate_asset.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert generate_asset._resolve_backend("auto") == "cuda"


def test_auto_fallback_order_keeps_full_resolution_first():
    assert list(generate_asset._attempt_schedule("auto", None, 800_000)) == [
        ("metal", None),
        ("kdtree", None),
        ("metal", 200_000),
        ("kdtree", 200_000),
    ]


def test_export_pre_simplifies_only_non_remesh_safety_attempts(monkeypatch, tmp_path: Path):
    simplify_calls = []
    export_calls = []

    fake_simplification = ModuleType("fast_simplification")

    def fake_simplify(vertices, faces, *, target_reduction):
        simplify_calls.append((len(faces), target_reduction))
        return vertices, faces[: generate_asset.SAFETY_FACE_TARGET]

    fake_simplification.simplify = fake_simplify
    monkeypatch.setitem(sys.modules, "fast_simplification", fake_simplification)

    class Exported:
        def export(self, path):
            path.write_bytes(b"candidate")

    def fake_to_glb(**kwargs):
        export_calls.append(
            {
                "faces": int(kwargs["faces"].shape[0]),
                "remesh": kwargs["remesh"],
                "remesh_band": kwargs["remesh_band"],
                "remesh_project": kwargs["remesh_project"],
                "remesh_project_max_dist": kwargs["remesh_project_max_dist"],
                "remesh_project_min_agreement": kwargs[
                    "remesh_project_min_agreement"
                ],
            }
        )
        return Exported()

    fake_postprocess = ModuleType("postprocess")
    fake_postprocess.to_glb = fake_to_glb
    fake_o_voxel = ModuleType("o_voxel")
    fake_o_voxel.postprocess = fake_postprocess
    monkeypatch.setitem(sys.modules, "o_voxel", fake_o_voxel)
    monkeypatch.setattr(generate_asset, "_metal_baker_available", lambda: (True, ""))

    face_count = generate_asset.SAFETY_FACE_TARGET + 1
    mesh = SimpleNamespace(
        vertices=torch.zeros((3, 3), dtype=torch.float32),
        faces=torch.zeros((face_count, 3), dtype=torch.int64),
        attrs=torch.zeros((1, 1), dtype=torch.float32),
        coords=torch.zeros((1, 3), dtype=torch.int64),
        layout={},
        voxel_size=1.0,
    )

    _, remesh_pre_simplified = generate_asset._export_pbr(
        mesh,
        tmp_path / "remesh.glb",
        baker="metal",
        target=generate_asset.SAFETY_FACE_TARGET,
        texture_size=512,
        remesh=True,
        remesh_band=1.25,
        remesh_project=0.2,
        remesh_project_max_dist=2.0,
        remesh_project_min_agreement=0.6,
        technical_safety_target=True,
    )
    _, legacy_pre_simplified = generate_asset._export_pbr(
        mesh,
        tmp_path / "legacy.glb",
        baker="metal",
        target=generate_asset.SAFETY_FACE_TARGET,
        texture_size=512,
        remesh=False,
        remesh_band=1.0,
        remesh_project=0.0,
        remesh_project_max_dist=1.5,
        remesh_project_min_agreement=0.5,
        technical_safety_target=True,
    )

    assert remesh_pre_simplified is False
    assert legacy_pre_simplified is True
    assert len(simplify_calls) == 1
    assert export_calls == [
        {
            "faces": face_count,
            "remesh": True,
            "remesh_band": 1.25,
            "remesh_project": 0.2,
            "remesh_project_max_dist": 2.0,
            "remesh_project_min_agreement": 0.6,
        },
        {
            "faces": generate_asset.SAFETY_FACE_TARGET,
            "remesh": False,
            "remesh_band": 1.0,
            "remesh_project": 0.0,
            "remesh_project_max_dist": 1.5,
            "remesh_project_min_agreement": 0.5,
        },
    ]


def test_simulated_bvh_failure_uses_full_kdtree(tmp_path: Path):
    calls = []

    def fake_export(
        mesh,
        path,
        *,
        baker,
        target,
        texture_size,
        remesh,
        remesh_band,
        remesh_project,
        remesh_project_max_dist,
        remesh_project_min_agreement,
        technical_safety_target,
    ):
        calls.append(
            (
                baker,
                target,
                texture_size,
                remesh,
                remesh_band,
                remesh_project,
                remesh_project_max_dist,
                remesh_project_min_agreement,
                technical_safety_target,
            )
        )
        if baker == "metal":
            raise RuntimeError("simulated Metal BVH failure")
        path.write_bytes(b"candidate")
        return object(), False

    _, chosen, attempts = generate_asset._run_pbr_attempts(
        _mesh(800_000),
        tmp_path / "candidate_pbr.glb",
        preferred_baker="auto",
        requested_target=None,
        texture_size=1024,
        remesh_band=1.25,
        remesh_project=0.2,
        remesh_project_max_dist=2.25,
        remesh_project_min_agreement=0.75,
        export_fn=fake_export,
    )

    assert calls == [
        ("metal", None, 1024, True, 1.25, 0.2, 2.25, 0.75, False),
        ("kdtree", None, 1024, True, 1.25, 0.2, 2.25, 0.75, False),
    ]
    assert chosen["baker"] == "kdtree"
    assert chosen["target_faces"] is None
    assert chosen["remeshed"] is False
    assert chosen["remesh_band"] == 1.25
    assert chosen["remesh_project"] == 0.2
    assert chosen["remesh_project_max_dist"] == 2.25
    assert chosen["remesh_project_min_agreement"] == 0.75
    assert [attempt["status"] for attempt in attempts] == ["failed", "ok"]
    assert [attempt["remeshed"] for attempt in attempts] == [False, False]


def test_safety_candidate_is_only_after_both_full_attempts(tmp_path: Path):
    calls = []

    def fake_export(
        mesh,
        path,
        *,
        baker,
        target,
        texture_size,
        remesh,
        remesh_band,
        remesh_project,
        remesh_project_max_dist,
        remesh_project_min_agreement,
        technical_safety_target,
    ):
        calls.append((baker, target, remesh, technical_safety_target))
        if target is None:
            raise RuntimeError("full-resolution baker failure")
        path.write_bytes(b"candidate")
        return object(), bool(not remesh and technical_safety_target)

    _, chosen, _ = generate_asset._run_pbr_attempts(
        _mesh(800_000),
        tmp_path / "candidate_pbr.glb",
        preferred_baker="auto",
        requested_target=None,
        texture_size=512,
        export_fn=fake_export,
    )

    assert calls == [
        ("metal", None, True, False),
        ("kdtree", None, True, False),
        ("metal", 200_000, True, True),
    ]
    assert chosen["technical_safety_target"] is True
    assert chosen["pre_simplified_before_bvh"] is False
    assert chosen["remeshed"] is True


def test_non_remesh_safety_attempt_keeps_pre_simplification(tmp_path: Path):
    calls = []

    def fake_export(
        mesh,
        path,
        *,
        baker,
        target,
        texture_size,
        remesh,
        remesh_band,
        remesh_project,
        remesh_project_max_dist,
        remesh_project_min_agreement,
        technical_safety_target,
    ):
        calls.append((baker, target, remesh, technical_safety_target))
        if target is None:
            raise RuntimeError("full-resolution baker failure")
        path.write_bytes(b"candidate")
        return object(), bool(not remesh and technical_safety_target)

    _, chosen, attempts = generate_asset._run_pbr_attempts(
        _mesh(800_000),
        tmp_path / "candidate_pbr.glb",
        preferred_baker="kdtree",
        requested_target=None,
        texture_size=512,
        remesh=False,
        export_fn=fake_export,
    )

    assert calls == [
        ("kdtree", None, False, False),
        ("kdtree", 200_000, False, True),
    ]
    assert chosen["pre_simplified_before_bvh"] is True
    assert [attempt["remeshed"] for attempt in attempts] == [False, False]


def test_explicit_kdtree_never_loads_metal():
    assert list(generate_asset._attempt_schedule("kdtree", None, 1000)) == [("kdtree", None)]


def test_explicit_200k_target_is_not_mislabeled_as_safety_fallback(tmp_path: Path):
    def fake_export(
        mesh,
        path,
        *,
        baker,
        target,
        texture_size,
        remesh,
        remesh_band,
        remesh_project,
        remesh_project_max_dist,
        remesh_project_min_agreement,
        technical_safety_target,
    ):
        path.write_bytes(b"candidate")
        return object(), False

    _, chosen, _ = generate_asset._run_pbr_attempts(
        _mesh(800_000),
        tmp_path / "candidate_pbr.glb",
        preferred_baker="kdtree",
        requested_target=200_000,
        texture_size=512,
        export_fn=fake_export,
    )

    assert chosen["target_faces"] == 200_000
    assert chosen["technical_safety_target"] is False


@pytest.mark.parametrize(
    ("remeshed", "boundary_edges", "winding_consistent", "expected"),
    [
        (True, 8, True, True),
        (False, 0, True, False),
        (True, 9, True, False),
        (True, 0, False, False),
    ],
)
def test_integrity_metadata_and_promotable_gate(
    monkeypatch,
    tmp_path: Path,
    remeshed: bool,
    boundary_edges: int,
    winding_consistent: bool,
    expected: bool,
):
    raw_path = tmp_path / "raw_full.glb"
    candidate_path = tmp_path / "candidate_pbr.glb"
    candidate_integrity = {
        "boundary_edges": boundary_edges,
        "winding_consistent": winding_consistent,
    }
    raw_integrity = {"boundary_edges": 12, "winding_consistent": False}
    measured = []

    def fake_measure_glb(path):
        measured.append(path)
        return candidate_integrity if path == candidate_path else raw_integrity

    monkeypatch.setattr(generate_asset, "measure_glb", fake_measure_glb)
    metadata = {
        "raw_full": {},
        "candidate_pbr": {"remeshed": remeshed},
    }

    generate_asset._record_integrity_metadata(metadata, raw_path, candidate_path)

    assert measured == [candidate_path, raw_path]
    assert metadata["candidate_pbr"]["integrity"] is candidate_integrity
    assert metadata["raw_full"]["integrity"] is raw_integrity
    assert metadata["candidate_pbr"]["promotable"] is expected


def test_raw_export_preserves_full_topology_and_writes_vertex_normals(tmp_path: Path):
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ]
    )
    output = tmp_path / "raw_full.glb"
    stats = generate_asset._export_raw(SimpleNamespace(vertices=vertices, faces=faces), output)

    from trellis2.gltf_validation import inspect_glb

    validated = inspect_glb(output, require_pbr=False)
    assert stats["vertices"] == validated["vertices"] == 8
    assert stats["triangles"] == validated["triangles"] == 12
