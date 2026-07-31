from pathlib import Path

from trellis2.model_revisions import (
    DINOV3_REPO,
    DINOV3_REVISION,
    MODEL_FILES,
    MODEL_REVISIONS,
    RMBG_REPO,
    RMBG_REVISION,
    SOURCE_REVISIONS,
    TRELLIS_REPO,
    TRELLIS_REVISION,
    revision_for_repo,
)

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_revisions_are_full_commits():
    assert MODEL_REVISIONS[TRELLIS_REPO] == TRELLIS_REVISION
    assert MODEL_REVISIONS[DINOV3_REPO] == DINOV3_REVISION
    assert MODEL_REVISIONS[RMBG_REPO] == RMBG_REVISION
    assert all(len(revision) == 40 for revision in MODEL_REVISIONS.values())
    assert all(len(revision) == 40 for revision in SOURCE_REVISIONS.values())
    assert MODEL_FILES.keys() == MODEL_REVISIONS.keys()
    assert all(files for files in MODEL_FILES.values())
    assert all(len(files) == len(set(files)) for files in MODEL_FILES.values())


def test_unknown_repo_keeps_explicit_revision():
    assert revision_for_repo("example/private-model", "release-1") == "release-1"


def test_source_revisions_are_present_in_reproducible_inputs():
    reproducibility_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("scripts/setup_macos.sh", "requirements_macos_core.txt", "README.md")
    )
    for revision in SOURCE_REVISIONS.values():
        assert revision in reproducibility_text
