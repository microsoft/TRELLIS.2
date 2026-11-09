"""Comparison table over the assets that every given method has results for.

    python baselines/compare.py results/dataset/all80_trellis_s0 results/dataset/sub10_material_anything \
        results/dataset/sub10_supermat

Reads metrics_uv.json / metrics_relit.json of each results folder (written by evaluate_uv.py /
evaluate_relit.py), keeps the assets present in all of them and prints a markdown table with
the input-as-albedo reference first. --names gives display names in the same order.
"""
import argparse
import json
from pathlib import Path

import numpy as np

UV = [("basecolor", "mae"), ("roughness", "mae"), ("metallic", "mae"),
      ("basecolor", "psnr"), ("roughness", "psnr"), ("metallic", "psnr")]
RELIT = ["mae", "psnr", "ssim", "lpips"]


def load(folder):
    out = {}
    for kind in ["uv", "relit"]:
        f = Path(folder) / f"metrics_{kind}.json"
        out[kind] = json.loads(f.read_text())["assets"] if f.exists() else {}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args()
    names = args.names or [Path(r).name for r in args.results]
    data = [load(r) for r in args.results]
    common = set.intersection(*[set(d["uv"]) & set(d["relit"]) for d in data])
    if not common:
        raise SystemExit("no asset has UV and relighting metrics for every method")
    common = sorted(common)

    rows = []
    ref = [data[0]["uv"][a].get("input_as_albedo") for a in common], [data[0]["relit"][a].get("input_as_albedo") for a in common]
    if all(ref[0]) and all(ref[1]):
        rows.append(("Input as albedo", ref[0], ref[1]))
    for n, d in zip(names, data):
        rows.append((n, [d["uv"][a]["pred"] for a in common], [d["relit"][a]["pred"] for a in common]))

    print(f"{len(common)} assets: " + ", ".join(a[:8] + "…" + a.split("__")[-1] for a in common) + "\n")
    head = [f"{c} {m.upper()}" for c, m in UV] + [f"relit {m.upper()}" for m in RELIT]
    print("| method | " + " | ".join(head) + " |")
    print("|---|" + "---|" * len(head))
    for n, uv, rl in rows:
        cells = [np.mean([u[c][m] for u in uv]) for c, m in UV] + [np.mean([r[m] for r in rl]) for m in RELIT]
        print(f"| {n} | " + " | ".join(f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}" for x in cells) + " |")


if __name__ == "__main__":
    main()
