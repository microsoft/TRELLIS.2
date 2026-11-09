"""Per-channel material metrics in UV space against the ground-truth PBR asset.

For every predicted asset (a folder written by unguided_trellis2.py: tex<k>_{basecolor,
roughness,metallic}.png + meta.json) the matching ground truth <base>_pbr.glb is loaded,
its maps are read per texture atlas (base color, roughness = G and metallic = B of the
metallicRoughness texture, times the material factors) and compared on the texels that
the UV layout covers. All values are in [0, 1]; base color stays in texture (sRGB) space.

    python baselines/evaluate_uv.py --pred results/unguided_trellis2 --gt-dir examples

Also reports the trivial input-as-albedo reference (baked texture as base color,
roughness --ref-roughness, metallic --ref-metallic) when the baked asset is found.
Writes <pred>/metrics_uv.json and prints a table.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ATTN_BACKEND", "xformers")

import numpy as np
import torch
from PIL import Image

import nvdiffrast.torch as dr
from unguided_trellis2 import load_parts

CHANNELS = ["basecolor", "roughness", "metallic"]


def locate(meta_path, gt_dir=None, gt_pattern="{base}_pbr.glb"):
    """(baked input, ground truth, HDR the input was baked with or None) for one result folder.
    Supports <base>_baked.glb next to <base>_pbr.glb (+ optional <base>_bake.json) and the
    dataset layout <object>/<hdr>_rot<deg>.glb next to <object>/pbr.glb + meta.json."""
    inp = Path(json.loads(Path(meta_path).read_text())["input"])
    base = inp.stem[:-len("_baked")] if inp.stem.endswith("_baked") else inp.stem
    folder = Path(gt_dir) if gt_dir else inp.parent
    gt = next((p for p in [folder / gt_pattern.format(base=base), folder / "pbr.glb"] if p.exists()), None)
    env = None
    if (folder / f"{base}_bake.json").exists():
        env = json.loads((folder / f"{base}_bake.json").read_text()).get("envmap")
    elif (inp.parent / "meta.json").exists():
        entries = json.loads((inp.parent / "meta.json").read_text())
        hit = [e for e in entries if isinstance(e, dict) and e.get("name") == inp.stem]
        env = Path(hit[0]["env_map"]).stem if hit and hit[0].get("env_map") else None
    return inp, gt, env


def gt_maps(path):
    """{frozenset(part names): {"basecolor", "roughness", "metallic", "mask"}} per texture atlas."""
    parts = load_parts(path)
    atlases = {}
    for p in parts:
        atlases.setdefault(p["tex_key"], []).append(p)
    out = {}
    for group in atlases.values():
        mat = group[0]["mesh"].visual.material
        bc = group[0]["texture"]  # base color texture x factor, float RGB
        h, w = bc.shape[:2]
        mr = getattr(mat, "metallicRoughnessTexture", None)
        rf = getattr(mat, "roughnessFactor", None)
        mf = getattr(mat, "metallicFactor", None)
        rf = 1.0 if rf is None else float(rf)  # glTF defaults
        mf = 1.0 if mf is None else float(mf)
        if mr is not None:
            mr = np.asarray(mr.convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255
            rough, metal = mr[..., 1] * rf, mr[..., 2] * mf
        else:
            rough, metal = np.full((h, w), rf, np.float32), np.full((h, w), mf, np.float32)
        mask = np.zeros((h, w), bool)
        for p in group:
            mask |= uv_coverage(p["uv"], np.asarray(p["mesh"].faces), h, w)
        out[frozenset(p["name"] for p in group)] = {
            "basecolor": bc, "roughness": rough, "metallic": metal, "mask": mask}
    return out


@torch.no_grad()
def uv_coverage(uv, faces, h, w):
    """Texels covered by the UV layout, with the same per-triangle tile shift as the bake."""
    uv_tri = torch.tensor(uv, device="cuda")[torch.tensor(faces, device="cuda")]
    uv_tri = (uv_tri - torch.floor(uv_tri.mean(1, keepdim=True))).reshape(-1, 2)
    f = torch.arange(len(uv_tri), dtype=torch.int32, device="cuda").reshape(-1, 3)
    clip = torch.cat([uv_tri[:, :1] * 2 - 1, (1 - uv_tri[:, 1:]) * 2 - 1,
                      torch.zeros_like(uv_tri[:, :1]), torch.ones_like(uv_tri[:, :1])], -1)[None]
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), clip, f, (h, w))
    return (rast[0, ..., 3] > 0).cpu().numpy()


def load_gray(path, h, w):
    im = Image.open(path).convert("L")
    if im.size != (w, h):
        im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255


def load_rgb(path, h, w):
    im = Image.open(path).convert("RGB")
    if im.size != (w, h):
        im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255


def metrics(pred, gt, mask):
    d = (pred - gt)[mask]
    mae = float(np.abs(d).mean())
    mse = float((d ** 2).mean())
    return {"mae": mae, "psnr": float(10 * np.log10(1.0 / max(mse, 1e-10)))}


def evaluate_asset(pred_dir, gt_path, baked_path, args):
    meta = json.loads((pred_dir / "meta.json").read_text())
    gts = gt_maps(gt_path)
    pred_rows, ref_rows, weights = [], [], []
    # part names of the baked input and the ground truth can differ (e.g. "x.001" vs "x_1"),
    # so parts are matched by geometry
    baked_parts = load_parts(baked_path)
    names = match_parts(baked_parts, load_parts(gt_path))
    baked = {frozenset(names[p["name"]] for p in g): g[0]["texture"] for g in _group(baked_parts)}
    for atlas in meta["atlases"]:
        key = frozenset(names.get(n, n) for n in atlas["parts"])
        gt = gts.get(key)
        if gt is None:
            raise ValueError(f"no ground-truth atlas with parts {sorted(key)} in {gt_path}")
        h, w = gt["mask"].shape
        a = atlas["index"]
        pred = {"basecolor": load_rgb(pred_dir / f"tex{a}_basecolor.png", h, w),
                "roughness": load_gray(pred_dir / f"tex{a}_roughness.png", h, w),
                "metallic": load_gray(pred_dir / f"tex{a}_metallic.png", h, w)}
        pred_rows.append({c: metrics(pred[c], gt[c], gt["mask"]) for c in CHANNELS})
        weights.append(int(gt["mask"].sum()))
        if baked is not None and key in baked:
            tex = baked[key]
            if tex.shape[:2] != (h, w):
                tex = np.asarray(Image.fromarray((tex * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                                 np.float32) / 255
            ref = {"basecolor": tex, "roughness": np.full((h, w), args.ref_roughness, np.float32),
                   "metallic": np.full((h, w), args.ref_metallic, np.float32)}
            ref_rows.append({c: metrics(ref[c], gt[c], gt["mask"]) for c in CHANNELS})
    res = {"pred": _weighted(pred_rows, weights)}
    if ref_rows and len(ref_rows) == len(pred_rows):
        res["input_as_albedo"] = _weighted(ref_rows, weights)
    return res


def match_parts(src, dst):
    """Map part names of src to dst: same face count, closest face-centroid mean and extent
    (robust to different vertex splitting on export); the name is the fallback."""
    def sig(p):
        m = p["mesh"]
        return len(m.faces), np.concatenate([m.triangles_center.mean(0), np.ptp(m.vertices, 0)])
    dst_sig = [(p["name"], *sig(p)) for p in dst]
    scale = max(float(np.ptp(np.concatenate([np.asarray(p["mesh"].vertices) for p in dst]), 0).max()), 1e-8)
    out = {}
    for p in src:
        n_f, s = sig(p)
        cands = [(np.abs(s - ds).max(), name) for name, dn, ds in dst_sig if dn == n_f]
        best = min(cands, default=(np.inf, None))
        out[p["name"]] = best[1] if best[0] < 1e-2 * scale else p["name"]
    return out


def _group(parts):
    groups = {}
    for p in parts:
        groups.setdefault(p["tex_key"], []).append(p)
    return groups.values()


def _weighted(rows, weights):
    """Texel-weighted mean over the atlases of one asset (PSNR from the pooled MSE)."""
    w = np.asarray(weights, np.float64) / sum(weights)
    out = {}
    for c in CHANNELS:
        mae = float(sum(wi * r[c]["mae"] for wi, r in zip(w, rows)))
        mse = float(sum(wi * 10 ** (-r[c]["psnr"] / 10) for wi, r in zip(w, rows)))
        out[c] = {"mae": mae, "psnr": float(10 * np.log10(1.0 / max(mse, 1e-10)))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="results folder of one method (one subfolder per asset)")
    ap.add_argument("--gt-dir", default=None, help="folder with the ground-truth and baked assets "
                    "(default: the folder of each baked input)")
    ap.add_argument("--gt-pattern", default="{base}_pbr.glb", help="falls back to pbr.glb in the same folder")
    ap.add_argument("--ref-roughness", type=float, default=0.5, help="input-as-albedo reference")
    ap.add_argument("--ref-metallic", type=float, default=0.0, help="input-as-albedo reference")
    args = ap.parse_args()

    pred_root = Path(args.pred)
    results = {}
    for meta_path in sorted(pred_root.glob("*/meta.json")):
        pred_dir = meta_path.parent
        inp, gt_path, _ = locate(meta_path, args.gt_dir, args.gt_pattern)
        if gt_path is None:
            print(f"skip {pred_dir.name}: no ground truth next to {inp}")
            continue
        try:
            results[pred_dir.name] = evaluate_asset(pred_dir, gt_path, inp, args)
        except Exception as e:
            print(f"FAILED {pred_dir.name}: {type(e).__name__}: {e}")
    if not results:
        sys.exit("nothing evaluated")

    summary = {}
    for method in ["pred", "input_as_albedo"]:
        rows = [r[method] for r in results.values() if method in r]
        if rows:
            summary[method] = {c: {m: float(np.mean([r[c][m] for r in rows])) for m in ["mae", "psnr"]}
                               for c in CHANNELS}
            summary[method]["n_assets"] = len(rows)
    (pred_root / "metrics_uv.json").write_text(json.dumps({"summary": summary, "assets": results}, indent=2))

    print(f"\n{pred_root}  ({len(results)} assets, mean over assets)")
    print(f"{'method':<18}" + "".join(f"{c + ' MAE':>16}{c + ' PSNR':>16}" for c in CHANNELS))
    for method, s in summary.items():
        name = pred_root.name if method == "pred" else "input-as-albedo"
        print(f"{name:<18}" + "".join(f"{s[c]['mae']:>16.4f}{s[c]['psnr']:>16.2f}" for c in CHANNELS))


if __name__ == "__main__":
    main()
