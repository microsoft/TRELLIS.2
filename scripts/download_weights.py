#!/usr/bin/env python3
"""Populate the dedicated Hugging Face cache with every pinned runtime input."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The Xet transport has produced non-resumable response-body decode failures
# on large model shards on macOS. Plain HTTP keeps .incomplete files in the
# selected HF cache and resumes them on the next attempt.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2.model_revisions import MODEL_FILES, MODEL_REVISIONS, TRELLIS_REPO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "trellis2" / "huggingface",
    )
    parser.add_argument("--offline", action="store_true", help="verify that the pinned cache is complete")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshots = {}
    for repo_id, revision in MODEL_REVISIONS.items():
        for attempt in range(1, args.retries + 1):
            try:
                snapshots[repo_id] = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=str(cache_dir),
                    local_files_only=args.offline,
                    max_workers=args.max_workers,
                    allow_patterns=list(MODEL_FILES[repo_id]),
                )
                break
            except GatedRepoError as exc:
                raise RuntimeError(
                    f"access to {repo_id}@{revision} is gated; accept its terms and run "
                    "`hf auth login`, then retry"
                ) from exc
            except (OSError, RuntimeError, HfHubHTTPError) as exc:
                if args.offline or attempt == args.retries:
                    raise RuntimeError(
                        f"failed to cache {repo_id}@{revision} after {attempt} attempt(s): {exc}"
                    ) from exc
                delay = min(5, attempt * 2)
                print(
                    f"Retrying {repo_id}@{revision} after transient download error "
                    f"({attempt}/{args.retries}): {exc}",
                    file=sys.stderr,
                )
                time.sleep(delay)

    print(
        json.dumps(
            {
                "cache_dir": str(cache_dir),
                "primary_repo": TRELLIS_REPO,
                "revisions": MODEL_REVISIONS,
                "runtime_files": MODEL_FILES,
                "snapshots": snapshots,
                "offline": args.offline,
                "max_workers": args.max_workers,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
