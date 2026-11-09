"""Qualitative comparison figure from the relighting renders of evaluate_relit.py.

    python baselines/figure.py --out fig.png --methods results/a results/b --names A B --assets N

Each row is one asset (rendered under a novel environment map from the same view for every
method): input render | ground truth | input-as-albedo | method 1 | method 2 | ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--assets", type=int, default=5, help="number of rows")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    roots = [Path(m) for m in args.methods]
    common = sorted(set.intersection(*[{p.parent.parent.name for p in r.glob(f"*/relit/view{args.view}_*.png")}
                                       for r in roots]))[:args.assets]
    t = args.tile
    header = ["input", "ground truth", "input as albedo"] + args.names
    rows = []
    for a in common:
        envs = sorted(set.intersection(*[{p.stem.split("_", 1)[1] for p in (r / a / "relit").glob(f"view{args.view}_*.png")}
                                         for r in roots]))
        env = envs[0]
        cells = []
        inp = Path(json.loads((roots[0] / a / "meta.json").read_text())["input"])
        cond = roots[0] / a / "cond.png"
        src = cond if cond.exists() else inp.with_suffix(".png")
        im = Image.open(src).convert("RGBA").resize((t, t))
        bg = Image.new("RGB", (t, t), (0, 0, 0))
        bg.paste(im, mask=im.split()[3])
        cells.append(np.asarray(bg))
        for k, r in enumerate(roots):
            strip = np.asarray(Image.open(r / a / "relit" / f"view{args.view}_{env}.png").convert("RGB"))
            w = strip.shape[1] // 3  # gt | prediction | input-as-albedo
            tiles = [np.asarray(Image.fromarray(strip[:, i * w:(i + 1) * w]).resize((t, t))) for i in range(3)]
            if k == 0:
                cells += [tiles[0], tiles[2]]
            cells.append(tiles[1])
        rows.append(np.concatenate(cells, 1))
    grid = Image.fromarray(np.concatenate(rows, 0))
    canvas = Image.new("RGB", (grid.width, grid.height + 24), (255, 255, 255))
    canvas.paste(grid, (0, 24))
    draw = ImageDraw.Draw(canvas)
    for i, h in enumerate(header):
        draw.text((i * t + 6, 5), h, fill=(0, 0, 0))
    canvas.save(args.out)
    print(f"{len(common)} rows -> {args.out}")


if __name__ == "__main__":
    main()
