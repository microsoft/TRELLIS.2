#!/bin/bash
# Unguided TRELLIS.2 baseline on a dataset + both evaluations, resumable.
#
#   bash baselines/run_unguided.sh <dataset dir> [resolution=1024] [results dir]
#
# Expects <base>_baked.glb next to <base>_pbr.glb (any folder depth).
set -e
cd "$(dirname "$0")/.."
DATA=${1:?usage: run_unguided.sh <dataset dir> [resolution] [results dir]}
RES=${2:-1024}
OUT=${3:-results/unguided_trellis2_r$RES}
PY=${PY:-python}
export ATTN_BACKEND=xformers

$PY baselines/unguided_trellis2.py --input "$DATA" --pattern "**/*_baked.glb" --out "$OUT" --resolution "$RES"
$PY baselines/evaluate_uv.py --pred "$OUT"
$PY baselines/evaluate_relit.py --pred "$OUT"
