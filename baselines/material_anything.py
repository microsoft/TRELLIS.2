"""Material Anything baseline on assets with baked lighting.

Runs Material Anything (https://github.com/3DTopia/MaterialAnything) in its mode for meshes
that already have a texture: the baked asset is written as mesh.obj + texture_kd.png, MA
renders it from 36 views, estimates albedo / roughness / metallic per view with its
material estimator (lighting-aware, with a confidence mask), back-projects them into the
mesh's UV layout and refines them in UV space. MA keeps the input UVs, so the result maps
are already in the original layout. They are written in the same format as
unguided_trellis2.py, so evaluate_uv.py / evaluate_relit.py work unchanged.

    python baselines/material_anything.py --input pbr_dataset/demo --out results/material_anything \
        --ma-dir path/to/MaterialAnything --ma-python path/to/ma_env/bin/python

Settings follow MA's bash/test.sh (36 views, UV refinement, --device 2080 i.e. 1024^2 UV
maps), plus --skip_controlnet / --no_update (see MA_ARGS; --skip_controlnet comes from
patches/material_anything_skip_controlnet.patch, apply it in the MA repo with git apply; it
also removes a crash on an undefined project_mask_image). Several texture atlases are packed side by side into one texture for
MA (like its own --use_multiple_objects) and split again afterwards.

Runs under the trellis2 env; MA itself runs in its own env (--ma-python: diffusers 0.28,
pytorch3d, kaolin) as a subprocess.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ATTN_BACKEND", "xformers")

import numpy as np
import trimesh
from PIL import Image

from unguided_trellis2 import load_parts, dilate, to_u8
from evaluate_uv import uv_coverage

MA_ARGS = ["--add_view_to_prompt", "--ddim_steps", "50", "--new_strength", "1", "--update_strength", "0.5",
           "--view_threshold", "0.1", "--blend", "0", "--dist", "1.2", "--num_viewpoints", "36",
           "--viewpoint_mode", "predefined", "--use_principle", "--update_steps", "0",
           "--update_mode", "heuristic", "--post_process", "--device", "2080",
           "--use_objaverse",  # glTF is Y up, which is what MA's Objaverse mode assumes
           # Textured input: MA estimates materials from renders of texture_kd.png. The
           # ControlNet repaint only produces images that are discarded (skip it), and the
           # "update" step would repaint regions from the text prompt (disable it).
           "--skip_controlnet", "--no_update"]


def write_ma_input(parts, atlases, in_dir, size):
    """mesh.obj (original coordinates, per-corner UVs packed per atlas) + texture_kd.png."""
    k_atlas = len(atlases)
    tex = np.zeros((size, size * k_atlas, 3), np.uint8)
    lines, vts, n_v = ["mtllib mesh.mtl", "usemtl material_0"], [], 0
    faces_out = []
    for a, idx in enumerate(atlases.values()):
        img = Image.fromarray(to_u8(parts[idx[0]]["texture"])).resize((size, size), Image.BILINEAR)
        tex[:, a * size:(a + 1) * size] = np.asarray(img)
        for k in idx:
            p = parts[k]
            v, f, uv = np.asarray(p["mesh"].vertices), np.asarray(p["mesh"].faces), p["uv"]
            lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
            uv_tri = uv[f]  # [F, 3, 2], shift every triangle into the [0, 1] tile, then pack
            uv_tri = uv_tri - np.floor(uv_tri.mean(1, keepdims=True))
            uv_tri[..., 0] = (uv_tri[..., 0] + a) / k_atlas
            base_vt = len(vts)
            vts.extend(uv_tri.reshape(-1, 2))
            for j, tri in enumerate(f):
                c = base_vt + 3 * j
                faces_out.append(f"f {tri[0] + 1 + n_v}/{c + 1} {tri[1] + 1 + n_v}/{c + 2} {tri[2] + 1 + n_v}/{c + 3}")
            n_v += len(v)
    lines += [f"vt {u:.6f} {w:.6f}" for u, w in vts] + faces_out
    (in_dir / "mesh.obj").write_text("\n".join(lines) + "\n")
    (in_dir / "mesh.mtl").write_text("newmtl material_0\nKd 1 1 1\nmap_Kd texture_kd.png\n")
    Image.fromarray(tex).save(in_dir / "texture_kd.png")


def run_asset(path, out_dir, args):
    t0 = time.time()
    parts = load_parts(path)
    atlases = {}
    for k, p in enumerate(parts):
        atlases.setdefault(p["tex_key"], []).append(k)
    work = Path(args.work_dir) / out_dir.name
    shutil.rmtree(work, ignore_errors=True)
    in_dir, ma_out = work / "input", work / "output"
    in_dir.mkdir(parents=True)
    write_ma_input(parts, atlases, in_dir, args.texture_size)

    base = path.stem[:-len("_baked")] if path.stem.endswith("_baked") else path.stem
    prompt = args.prompt.format(name=base.replace("_", " "))
    cmd = [args.ma_python, "scripts/generate_texture_pbr_3d.py",
           "--image2materials_model", "./pretrained_models/material_estimator",
           "--uvrefine_model", "./pretrained_models/material_refiner",
           "--input_dir", str(in_dir), "--output_dir", str(ma_out),
           "--obj_name", "mesh", "--obj_file", "mesh.obj", "--prompt", prompt,
           "--seed", str(args.seed), *MA_ARGS]
    with open(out_dir / "ma_log.txt", "w") as log:
        ret = subprocess.run(cmd, cwd=args.ma_dir, stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONPATH": args.ma_dir})
    if ret.returncode != 0:
        raise RuntimeError(f"Material Anything failed (exit {ret.returncode}), see {out_dir / 'ma_log.txt'}")
    t_ma = time.time() - t0

    mat_dir = next(ma_out.rglob("final_albedo_after.png")).parent
    maps = {n: np.asarray(Image.open(mat_dir / f"final_{n}_after.png").convert("RGB" if n == "albedo" else "L"))
            for n in ["albedo", "roughness", "metallic"]}
    shutil.copy(in_dir / "texture_kd.png", out_dir / "ma_input_texture.png")

    k_atlas, materials, atlas_meta = len(atlases), {}, []
    for a, (key, idx) in enumerate(atlases.items()):
        h, w = parts[idx[0]]["texture"].shape[:2]
        if h * w <= 1:
            h = w = args.texture_size
        crop = lambda m: m[:, m.shape[1] * a // k_atlas:m.shape[1] * (a + 1) // k_atlas]
        res = lambda m: np.asarray(Image.fromarray(crop(m)).resize((w, h), Image.BILINEAR))
        bc, rough, metal = res(maps["albedo"]), res(maps["roughness"]), res(maps["metallic"])
        mask = np.zeros((h, w), bool)
        for k in idx:
            mask |= uv_coverage(parts[k]["uv"], np.asarray(parts[k]["mesh"].faces), h, w)
        Image.fromarray(bc).save(out_dir / f"tex{a}_basecolor.png")
        Image.fromarray(rough).save(out_dir / f"tex{a}_roughness.png")
        Image.fromarray(metal).save(out_dir / f"tex{a}_metallic.png")
        Image.fromarray(to_u8(mask.astype(np.float32))).save(out_dir / f"tex{a}_mask.png")
        rough_d, metal_d = dilate(rough, mask), dilate(metal, mask)
        materials[key] = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(dilate(bc, mask)),
            metallicRoughnessTexture=Image.fromarray(np.stack([np.zeros_like(rough_d), rough_d, metal_d], -1)),
            metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)
        atlas_meta.append({"index": a, "size": [h, w], "parts": [parts[k]["name"] for k in idx],
                           "coverage": round(float(mask.mean()), 4)})
    glb = []
    for p in parts:
        mesh = p["mesh"].copy()
        mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=materials[p["tex_key"]])
        glb.append(mesh)
    trimesh.Scene(glb).export(out_dir / "textured.glb")
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    meta = {"input": str(path), "method": "material_anything", "parts": [p["name"] for p in parts],
            "atlases": atlas_meta, "prompt": prompt, "seed": args.seed, "ma_args": MA_ARGS,
            "texture_size": args.texture_size, "time_s": {"material_anything": round(t_ma, 1)},
            "total_s": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="an asset file, or a directory searched with --pattern")
    ap.add_argument("--pattern", default="**/*_baked.glb")
    ap.add_argument("--out", default="results/material_anything")
    ap.add_argument("--ma-dir", required=True, help="clone of https://github.com/3DTopia/MaterialAnything, patched")
    ap.add_argument("--ma-python", required=True, help="python of the Material Anything environment")
    ap.add_argument("--work-dir", default="/tmp/material_anything_work", help="MA's intermediate renders (large)")
    ap.add_argument("--prompt", default="{name}", help="text prompt for MA, {name} = file name without _baked")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--texture-size", type=int, default=1024, help="size of each atlas inside MA's texture")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    assets = [inp] if inp.is_file() else sorted(inp.glob(args.pattern))
    if not assets:
        sys.exit(f"no assets found in {inp} with pattern {args.pattern}")
    for i, path in enumerate(assets):
        name = path.stem if inp.is_file() else str(path.relative_to(inp).with_suffix("")).replace(os.sep, "__")
        out_dir = Path(args.out) / name
        if (out_dir / "meta.json").exists() and not args.overwrite:
            print(f"[{i + 1}/{len(assets)}] skip {name} (done)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error.txt").unlink(missing_ok=True)
        try:
            meta = run_asset(path, out_dir, args)
            print(f"[{i + 1}/{len(assets)}] {name}: {meta['total_s']} s", flush=True)
        except Exception as e:
            print(f"[{i + 1}/{len(assets)}] FAILED {name}: {type(e).__name__}: {e}", flush=True)
            (out_dir / "error.txt").write_text(traceback.format_exc())


if __name__ == "__main__":
    main()
