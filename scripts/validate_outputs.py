#!/usr/bin/env python3
"""Validate a raw_full.glb/candidate_pbr.glb pair and print JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2.gltf_validation import validate_output_pair


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_full", type=Path)
    parser.add_argument("candidate_pbr", type=Path)
    args = parser.parse_args()
    report = validate_output_pair(args.raw_full.resolve(), args.candidate_pbr.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
