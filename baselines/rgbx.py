"""RGB<->X baseline (RGB->X, Zeng et al., SIGGRAPH 2024) on assets with baked lighting.

The baked asset is rendered (unlit, i.e. with the baked texture) from the 6 orthographic views used for
SuperMat, RGB->X predicts albedo / roughness / metallic for every view independently, and the predictions
are back-projected into the mesh's original UV layout (depth-tested, weighted by cos^2 between normal and
view direction). Besides the fused maps it writes the per-texel spread across views
(tex<k>_<channel>_sigma.png, standard deviation x 4, clipped), for the confidence weights of the proposal.

RGB->X is trained on interior scenes; this is the go/no-go test for using it as 2D material prior on
objects. Output format as unguided_trellis2.py, so evaluate_uv.py / evaluate_relit.py apply.

    python baselines/rgbx.py --input pbr_dataset --pattern "*/*_rot*.glb" --out results/rgbx \
        --rgbx-dir path/to/rgbx

Code and weights (zheng95z/rgbx, zheng95z/rgb-to-x) are under the Adobe Research License (noncommercial
research only).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import trimesh
from PIL import Image

from unguided_trellis2 import load_parts, normalization, normalize, to_u8, dilate
from supermat import VIEWS, EXTENT, render_views, texel_samples, back_project

PROMPTS = {"albedo": "Albedo (diffuse basecolor)", "roughness": "Roughness", "metallic": "Metallicness"}


def load_rgbx(rgbx_dir):
    sys.path.insert(0, str(Path(rgbx_dir) / "rgb2x"))
    from diffusers import DDIMScheduler
    from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained("zheng95z/rgb-to-x", torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True,
                                               timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def predict(pipe, rgba, size, steps, seed, background):
    """rgba: [H, W, 4] in [0, 1] (sRGB). Returns dict channel -> [size, size, C] in [0, 1]."""
    rgb, a = rgba[..., :3], rgba[..., 3:]
    rgb = rgb * a + background * (1 - a)
    photo = (rgb.permute(2, 0, 1) ** 2.2)  # the pipeline expects linear RGB (as load_ldr_image(from_srgb=True))
    photo = torch.nn.functional.interpolate(photo[None], (size, size), mode="bilinear", align_corners=False)[0]
    out = {}
    for ch, prompt in PROMPTS.items():
        g = torch.Generator(device="cuda").manual_seed(seed)
        img = pipe(prompt=prompt, photo=photo.half(), num_inference_steps=steps, height=size, width=size,
                   generator=g, required_aovs=[ch], output_type="np").images[0][0]
        img = torch.tensor(np.asarray(img), dtype=torch.float32, device="cuda")
        out[ch] = img if ch == "albedo" else img[..., :1]
    return out


def run_asset(pipe, path, out_dir, args):
    t0 = time.time()
    parts = load_parts(path)
    center, scale = normalization(np.concatenate([p["mesh"].vertices for p in parts]))
    rot = lambda n: np.stack([n[:, 0], -n[:, 2], n[:, 1]], 1).astype(np.float32)
    pv = []
    for p in parts:
        v = normalize(np.asarray(p["mesh"].vertices), center, scale)
        pv.append({"v": v * EXTENT, "f": np.asarray(p["mesh"].faces, np.int64), "uv": p["uv"], "texture": p["texture"],
                   "n": rot(np.asarray(p["mesh"].vertex_normals))})
    images, depths = render_views(pv, args.render_size)
    views = out_dir / "rgbx_views"
    views.mkdir(exist_ok=True)
    preds = []
    for k, im in enumerate(images):
        p = predict(pipe, im, args.size, args.steps, args.seed, args.background)
        p = {c: torch.nn.functional.interpolate(x.permute(2, 0, 1)[None], (args.render_size, args.render_size),
                                                mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
             for c, x in p.items()}
        preds.append(p)
        if k < args.save_views:
            Image.fromarray(to_u8(torch.cat([im[..., :3], p["albedo"], p["roughness"].expand(-1, -1, 3),
                                             p["metallic"].expand(-1, -1, 3)], 1).cpu().numpy())).save(views / f"view{k}.png")
    stacks = {c: [torch.cat([p[c], p[c] ** 2], -1) for p in preds] for c in PROMPTS}  # mean and second moment

    atlases = {}
    for k, p in enumerate(parts):
        atlases.setdefault(p["tex_key"], []).append(k)
    atlas_meta, mats = [], {}
    for a, (key, idx) in enumerate(atlases.items()):
        h, w = parts[idx[0]]["texture"].shape[:2]
        res = {c: np.zeros((h, w, 2 * (3 if c == "albedo" else 1)), np.float32) for c in PROMPTS}
        cover = np.zeros((h, w), bool); seen = np.zeros((h, w), bool)
        for k in idx:
            q = pv[k]
            mask, pos, nrm = texel_samples(q["uv"], q["f"], q["v"], q["n"], h, w)
            m = mask.cpu().numpy()
            for c in PROMPTS:
                vals, _, sn = back_project(stacks[c], depths, pos, nrm, mask, args.render_size, args.depth_tol)
                res[c][m] = vals[m]
                seen[m] |= sn[m]
            cover |= m
        out = {}
        for c, x in res.items():
            n = x.shape[-1] // 2
            mean, sq = x[..., :n], x[..., n:]
            sigma = np.sqrt(np.clip(sq - mean ** 2, 0, None)).mean(-1)
            out[c] = mean if n == 3 else mean[..., 0]
            Image.fromarray(to_u8(np.clip(sigma * 4, 0, 1))).save(out_dir / f"tex{a}_{c}_sigma.png")
        known = cover & seen
        Image.fromarray(to_u8(out["albedo"])).save(out_dir / f"tex{a}_basecolor.png")
        Image.fromarray(to_u8(out["roughness"])).save(out_dir / f"tex{a}_roughness.png")
        Image.fromarray(to_u8(out["metallic"])).save(out_dir / f"tex{a}_metallic.png")
        Image.fromarray(to_u8(cover.astype(np.float32))).save(out_dir / f"tex{a}_mask.png")
        fill = lambda img: dilate(img, known) if known.any() else img
        mats[key] = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(fill(to_u8(out["albedo"]))),
            metallicRoughnessTexture=Image.fromarray(np.stack(
                [np.zeros((h, w), np.uint8), fill(to_u8(out["roughness"])), fill(to_u8(out["metallic"]))], -1)),
            metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)
        atlas_meta.append({"index": a, "size": [h, w], "parts": [parts[k]["name"] for k in idx],
                           "seen_fraction": round(float(known.sum() / max(cover.sum(), 1)), 4)})
    glb = []
    for p in parts:
        mesh = p["mesh"].copy()
        mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=mats[p["tex_key"]])
        glb.append(mesh)
    trimesh.Scene(glb).export(out_dir / "textured.glb")
    meta = {"input": str(path), "method": "rgbx", "parts": [p["name"] for p in parts], "atlases": atlas_meta,
            "views": len(VIEWS), "size": args.size, "steps": args.steps, "seed": args.seed,
            "background": args.background, "total_s": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta["total_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--pattern", default="*/*_rot*.glb")
    ap.add_argument("--out", default="results/rgbx")
    ap.add_argument("--rgbx-dir", required=True, help="clone of https://github.com/zheng95z/rgbx")
    ap.add_argument("--only", default=None, help="comma-separated object id prefixes")
    ap.add_argument("--render-size", type=int, default=1024)
    ap.add_argument("--size", type=int, default=512, help="RGB->X resolution")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--background", type=float, default=0.0, help="gray level behind the object")
    ap.add_argument("--depth-tol", type=float, default=0.01)
    ap.add_argument("--save-views", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out_root = Path(args.out).resolve()
    assets = sorted(inp.glob(args.pattern))
    if args.only:
        assets = [p for p in assets if any(p.parent.name.startswith(o) for o in args.only.split(","))]
    pipe = load_rgbx(args.rgbx_dir)
    for path in assets:
        name = str(path.relative_to(inp).with_suffix("")).replace(os.sep, "__")
        out_dir = out_root / name
        if (out_dir / "meta.json").exists() and not args.overwrite:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            s = run_asset(pipe, path, out_dir, args)
            print(f"{name[:8]}…{path.stem}: {s} s", flush=True)
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)
            (out_dir / "error.txt").write_text(traceback.format_exc())
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
