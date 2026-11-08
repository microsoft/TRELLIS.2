"""Mean +- std over runs (e.g. seeds) of the metrics written by evaluate_uv.py / evaluate_relit.py.

    python baselines/summarize.py results/seeds/r1024_s*      # one results folder per run

Prints a markdown table (asset-averaged per run, then mean +- std over runs) for the method
and the input-as-albedo reference.
"""
import json
import sys
from pathlib import Path

import numpy as np

UV = [("basecolor", "mae"), ("basecolor", "psnr"), ("roughness", "mae"), ("roughness", "psnr"),
      ("metallic", "mae"), ("metallic", "psnr")]
RELIT = ["mae", "psnr", "ssim", "lpips"]


def fmt(vals):
    vals = np.asarray(vals, np.float64)
    return f"{vals.mean():.3f} ± {vals.std():.3f}" if len(vals) > 1 else f"{vals.mean():.3f}"


def main():
    runs = [Path(p) for p in sys.argv[1:] if Path(p).is_dir()]
    if not runs:
        sys.exit(__doc__)
    for method, label in [("pred", "method"), ("input_as_albedo", "input-as-albedo")]:
        uv = [json.loads((r / "metrics_uv.json").read_text())["summary"].get(method)
              for r in runs if (r / "metrics_uv.json").exists()]
        relit = [json.loads((r / "metrics_relit.json").read_text())["summary"].get(method)
                 for r in runs if (r / "metrics_relit.json").exists()]
        uv, relit = [u for u in uv if u], [r for r in relit if r]
        print(f"\n### {label} ({len(uv)} runs UV, {len(relit)} runs relit)\n")
        if uv:
            print("| " + " | ".join(f"{c} {m.upper()}" for c, m in UV) + " |")
            print("|" + "---|" * len(UV))
            print("| " + " | ".join(fmt([u[c][m] for u in uv]) for c, m in UV) + " |")
        if relit:
            print("\n| relit " + " | relit ".join(m.upper() for m in RELIT) + " |")
            print("|" + "---|" * len(RELIT))
            print("| " + " | ".join(fmt([r[m] for r in relit]) for m in RELIT) + " |")


if __name__ == "__main__":
    main()
