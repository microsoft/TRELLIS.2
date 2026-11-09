"""Relighting metrics: render prediction and ground truth under novel environment lighting.

Every method is given as a GLB with the same geometry as the ground truth (the predicted
asset written by a baseline, e.g. results/<method>/<asset>/textured.glb). Both are
rendered with the TRELLIS.2 PBR renderer (split-sum environment lighting, SSAO, ACES tone
mapping, gamma) from --views camera directions under every HDR in --envmaps, and compared
with MAE / PSNR (on object pixels), SSIM and LPIPS (on the full image, black background).
Normal maps are ignored for all methods (face normals), so only the materials differ.

    python baselines/evaluate_relit.py --pred results/unguided_trellis2 --gt-dir examples

Also renders the input-as-albedo reference (baked texture as base color, constant
roughness/metallic). Writes <pred>/metrics_relit.json and example renders to
<pred>/<asset>/relit/.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # before cv2 is imported anywhere
os.environ.setdefault("ATTN_BACKEND", "xformers")
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unguided_trellis2 import load_parts, normalization, normalize
from evaluate_uv import locate
from trellis2.renderers import PbrMeshRenderer, EnvMap
from trellis2.representations.mesh.base import MeshWithPbrMaterial, PbrMaterial, Texture
from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


def material_maps(mesh):
    """(base color [H,W,3] or None, base factor, roughness [H,W] or None, rough factor,
    metallic [H,W] or None, metal factor) from a trimesh PBR material, glTF defaults."""
    mat = mesh.visual.material
    bc_img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    bc_f = getattr(mat, "baseColorFactor", None)
    bc_f = [1.0, 1.0, 1.0] if bc_f is None else (np.asarray(bc_f[:3], np.float32) / 255).tolist()
    bc = None if bc_img is None else np.asarray(bc_img.convert("RGB"), np.float32) / 255
    mr_img = getattr(mat, "metallicRoughnessTexture", None)
    rf, mf = getattr(mat, "roughnessFactor", None), getattr(mat, "metallicFactor", None)
    rf = 1.0 if rf is None else float(rf)
    mf = 1.0 if mf is None else float(mf)
    rough = metal = None
    if mr_img is not None:
        mr = np.asarray(mr_img.convert("RGB"), np.float32) / 255
        rough, metal = mr[..., 1], mr[..., 2]
    return bc, bc_f, rough, rf, metal, mf


def glb_to_pbr_mesh(path, center, scale, const_rough=None, const_metal=None):
    """All parts of a GLB as one MeshWithPbrMaterial in TRELLIS space (one material per
    distinct texture). const_* replace roughness / metallic by a constant."""
    parts = load_parts(path)
    tex = lambda a: Texture(torch.tensor(a if a.ndim == 3 else a[..., None], device="cuda").contiguous())
    materials, mat_index = [], {}
    verts, faces, mat_ids, uvs, offset = [], [], [], [], 0
    for p in parts:
        key = p["tex_key"]
        if key not in mat_index:
            bc, bc_f, rough, rf, metal, mf = material_maps(p["mesh"])
            if const_rough is not None:
                rough, rf = None, const_rough
            if const_metal is not None:
                metal, mf = None, const_metal
            mat_index[key] = len(materials)
            materials.append(PbrMaterial(
                base_color_texture=None if bc is None else tex(bc), base_color_factor=bc_f,
                roughness_texture=None if rough is None else tex(rough), roughness_factor=rf,
                metallic_texture=None if metal is None else tex(metal), metallic_factor=mf,
            ).to("cuda"))
        f = np.asarray(p["mesh"].faces, np.int64)
        verts.append(normalize(np.asarray(p["mesh"].vertices), center, scale))
        faces.append(f + offset)
        offset += len(verts[-1])
        mat_ids.append(np.full(len(f), mat_index[key]))
        uv = p["uv"][f].copy()  # [F, 3, 2]
        uv[..., 1] = 1 - uv[..., 1]  # trimesh v points up, texture row 0 is the top
        uvs.append(uv)
    t = lambda a, dt: torch.tensor(np.concatenate(a), dtype=dt, device="cuda")
    return MeshWithPbrMaterial(t(verts, torch.float32), t(faces, torch.int32), t(mat_ids, torch.long),
                               t(uvs, torch.float32), materials)


def load_envmap(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return EnvMap(torch.tensor(cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB), dtype=torch.float32, device="cuda"))


class Scorer:
    def __init__(self):
        import lpips
        import kornia
        self.lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
        self.ssim = kornia.metrics.ssim

    @torch.no_grad()
    def __call__(self, pred, gt, mask):
        """pred, gt: [3, H, W] in [0, 1]; mask: [H, W] bool (object pixels)."""
        d = (pred - gt)[:, mask]
        mse = float((d ** 2).mean())
        return {"mae": float(d.abs().mean()),
                "psnr": float(10 * np.log10(1.0 / max(mse, 1e-10))),
                "ssim": float(self.ssim(pred[None], gt[None], 11).mean()),
                "lpips": float(self.lpips(pred[None] * 2 - 1, gt[None] * 2 - 1).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="results folder of one method (one subfolder per asset)")
    ap.add_argument("--gt-dir", default=None, help="folder with the ground-truth and baked assets "
                    "(default: the folder of each baked input)")
    ap.add_argument("--gt-pattern", default="{base}_pbr.glb", help="falls back to pbr.glb in the same folder")
    ap.add_argument("--pred-glb", default="textured.glb", help="file name of the prediction in each asset folder")
    ap.add_argument("--envmaps", default=str(Path(__file__).resolve().parents[1] / "env_maps"),
                    help="folder with .exr/.hdr environment maps")
    ap.add_argument("--views", type=int, default=4, help="camera yaws, evenly spaced")
    ap.add_argument("--pitch", type=float, default=20.0)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--ref-roughness", type=float, default=0.5)
    ap.add_argument("--ref-metallic", type=float, default=0.0)
    ap.add_argument("--save-renders", type=int, default=2, help="environment maps to save renders for")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    env_paths = sorted([*Path(args.envmaps).glob("*.exr"), *Path(args.envmaps).glob("*.hdr")])
    if not env_paths:
        sys.exit(f"no environment maps in {args.envmaps}")
    envmaps = {p.stem: load_envmap(p) for p in env_paths}
    yaws = [2 * np.pi * i / args.views for i in range(args.views)]
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, [np.radians(args.pitch)] * args.views, 2.0, 40.0)
    renderer = PbrMeshRenderer({"resolution": args.resolution, "near": 1, "far": 100, "ssaa": 2, "peel_layers": 4})
    scorer = Scorer()

    pred_root = Path(args.pred)
    results = {}
    for meta_path in sorted(pred_root.glob("*/meta.json")):
        pred_dir = meta_path.parent
        baked_path, gt_path, skip = locate(meta_path, args.gt_dir, args.gt_pattern)  # skip: HDR used for baking
        pred_path = pred_dir / args.pred_glb
        if gt_path is None or not pred_path.exists():
            print(f"skip {pred_dir.name}: missing ground truth or {pred_path}")
            continue
        center, scale = normalization(np.concatenate([p["mesh"].vertices for p in load_parts(gt_path)]))
        meshes = {"gt": glb_to_pbr_mesh(gt_path, center, scale), "pred": glb_to_pbr_mesh(pred_path, center, scale)}
        if baked_path.exists():
            meshes["input_as_albedo"] = glb_to_pbr_mesh(baked_path, center, scale, args.ref_roughness, args.ref_metallic)
        scores = {m: [] for m in meshes if m != "gt"}
        out_dir = pred_dir / "relit"
        out_dir.mkdir(exist_ok=True)
        for v, (e, i) in enumerate(zip(extr, intr)):
            renders = {m: renderer.render(mesh, e, i, envmaps) for m, mesh in meshes.items()}
            mask = renders["gt"].mask > 0.5
            for k, name in enumerate(envmaps):
                if name == skip:
                    continue
                gt_img = renders["gt"][f"shaded_{name}"].clamp(0, 1)
                for m in scores:
                    scores[m].append(scorer(renders[m][f"shaded_{name}"].clamp(0, 1), gt_img, mask))
                if k < args.save_renders:
                    row = [renders[m][f"shaded_{name}"].clamp(0, 1) for m in meshes]  # gt, pred, ref
                    img = (torch.cat(row, 2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img).save(out_dir / f"view{v}_{name}.png")
        results[pred_dir.name] = {m: {k: float(np.mean([s[k] for s in rows])) for k in rows[0]}
                                  for m, rows in scores.items()}
        print(pred_dir.name, json.dumps(results[pred_dir.name]), flush=True)
    if not results:
        sys.exit("nothing evaluated")

    summary = {}
    for m in ["pred", "input_as_albedo"]:
        rows = [r[m] for r in results.values() if m in r]
        if rows:
            summary[m] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            summary[m]["n_assets"] = len(rows)
    (pred_root / "metrics_relit.json").write_text(json.dumps(
        {"summary": summary, "assets": results, "envmaps": list(envmaps), "views": args.views,
         "pitch": args.pitch, "resolution": args.resolution}, indent=2))
    print(f"\n{pred_root}  ({len(results)} assets, {len(envmaps)} env maps x {args.views} views)")
    print(f"{'method':<30}{'MAE':>10}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}")
    for m, s in summary.items():
        name = pred_root.name if m == "pred" else "input-as-albedo"
        print(f"{name:<30}{s['mae']:>10.4f}{s['psnr']:>10.2f}{s['ssim']:>10.4f}{s['lpips']:>10.4f}")


if __name__ == "__main__":
    main()
