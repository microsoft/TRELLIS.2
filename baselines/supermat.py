"""SuperMat baseline on assets with baked lighting.

SuperMat (https://github.com/hyj542682306/SuperMat) estimates albedo / roughness / metallic
from images. Its 3D use is reproduced here: the baked asset is rendered (unlit, i.e. with
the baked texture) from the 6 orthographic views of SuperMat's multi-view example, the
multi-view model predicts materials per view, the predictions are back-projected into the
mesh's original UV layout (depth-tested, weighted by the cosine between normal and view
direction) and finally passed through SuperMat's UV refinement networks (base color and
roughness/metallic) with the UV position map and mask.

    python baselines/supermat.py --input pbr_dataset --pattern "*/*_rot*.glb" --out results/supermat \
        --sm-dir path/to/SuperMat --ckpt-dir path/to/SuperMat/checkpoints

Runs in an env with SuperMat's requirements (diffusers) on top of the trellis2 packages.
Runs in three passes over all assets so that only one network is loaded at a time.
Output format as unguided_trellis2.py, plus tex<k>_*_proj.png (before UV refinement).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
from PIL import Image

from unguided_trellis2 import load_parts, normalization, normalize, to_u8, dilate

# camera positions / up vectors of SuperMat's 6-view example (Z up, Blender camera convention)
VIEWS = [((0, -1, 0), (0, 0, 1)), ((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)),
         ((-1, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, -1, 0)), ((0, 0, -1), (0, 1, 0))]
EXTENT = 0.9  # object size in the orthographic frame [-0.5, 0.5] (as in the example renders)


def c2w_matrix(pos, up):
    z = np.asarray(pos, np.float64)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, pos
    return m


def to_view(points, c2w):
    """Camera-space x, y and depth (distance in front of the camera)."""
    w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=points.device)
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    return pc[:, 0], pc[:, 1], -pc[:, 2]


@torch.no_grad()
def render_views(parts_view, size):
    """RGBA renders of the baked texture and camera-space depth maps for the 6 views."""
    verts, faces, uvs, face_part, off = [], [], [], [], 0
    for k, p in enumerate(parts_view):
        verts.append(p["v"]); uvs.append(p["uv"]); faces.append(p["f"] + off)
        face_part.append(np.full(len(p["f"]), k)); off += len(p["v"])
    v = torch.tensor(np.concatenate(verts), device="cuda")
    f = torch.tensor(np.concatenate(faces), dtype=torch.int32, device="cuda")
    uv = torch.tensor(np.concatenate(uvs), device="cuda")
    face_part = torch.tensor(np.concatenate(face_part), device="cuda")
    ctx = dr.RasterizeCudaContext()
    images, depths = [], []
    for pos, up in VIEWS:
        c2w = c2w_matrix(pos, up)
        x, y, d = to_view(v, c2w)
        clip = torch.stack([x / 0.5, y / 0.5, d - 1.0, torch.ones_like(d)], -1)[None]
        rast, _ = dr.rasterize(ctx, clip, f, (size, size))
        tri = rast[0, ..., 3].long() - 1
        uv_px = dr.interpolate(uv[None], rast, f)[0][0]
        depth = dr.interpolate(d[None, :, None].contiguous(), rast, f)[0][0, ..., 0]
        rgb = torch.zeros(size, size, 3, device="cuda")
        for k, p in enumerate(parts_view):
            m = (tri >= 0) & (face_part[tri.clamp(min=0)] == k)
            if not m.any():
                continue
            tex = torch.tensor(p["texture"], device="cuda").permute(2, 0, 1)[None]
            uv_m = uv_px[m] % 1
            grid = torch.stack([uv_m[:, 0] * 2 - 1, (1 - uv_m[:, 1]) * 2 - 1], -1)[None, None]
            rgb[m] = F.grid_sample(tex, grid, mode="bilinear", align_corners=False)[0, :, 0].T
        alpha = (tri >= 0).float()
        depth = torch.where(tri >= 0, depth, torch.full_like(depth, 1e9))
        images.append(torch.cat([rgb, alpha[..., None]], -1).flip(0))  # GL rows are bottom-up
        depths.append(depth.flip(0))
    return images, depths


@torch.no_grad()
def texel_samples(uv, faces, verts, normals, h, w):
    """Positions and normals of the texels covered by the UV layout (per-triangle tile shift)."""
    f = torch.tensor(faces, device="cuda")
    uv_tri = torch.tensor(uv, device="cuda")[f]
    uv_tri = (uv_tri - torch.floor(uv_tri.mean(1, keepdim=True))).reshape(-1, 2)
    idx = torch.arange(len(uv_tri), dtype=torch.int32, device="cuda").reshape(-1, 3)
    clip = torch.cat([uv_tri[:, :1] * 2 - 1, (1 - uv_tri[:, 1:]) * 2 - 1,
                      torch.zeros_like(uv_tri[:, :1]), torch.ones_like(uv_tri[:, :1])], -1)[None]
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), clip, idx, (h, w))
    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(torch.tensor(verts, device="cuda")[f].reshape(-1, 3)[None], rast, idx)[0][0]
    nrm = dr.interpolate(torch.tensor(normals, device="cuda")[f].reshape(-1, 3)[None], rast, idx)[0][0]
    return mask, pos, F.normalize(nrm, dim=-1)


@torch.no_grad()
def back_project(maps, depths, pos, nrm, mask, size, depth_tol):
    """Weighted average of the per-view predictions at every covered texel.
    maps: per view [H, W, C] in [0, 1]; returns values [h, w, C], best cosine [h, w], seen [h, w]."""
    h, w = mask.shape
    p, n = pos[mask], nrm[mask]
    acc = torch.zeros(len(p), maps[0].shape[-1], device="cuda")
    wsum = torch.zeros(len(p), device="cuda")
    best = torch.zeros(len(p), device="cuda")
    for (cam, up), m, dmap in zip(VIEWS, maps, depths):
        c2w = c2w_matrix(cam, up)
        x, y, d = to_view(p, c2w)
        to_cam = torch.tensor(cam, dtype=torch.float32, device="cuda")
        cos = (n @ (to_cam / to_cam.norm())).clamp(min=0)
        px = ((x / 0.5 + 1) / 2 * size).long().clamp(0, size - 1)
        py = ((1 - (y / 0.5 + 1) / 2) * size).long().clamp(0, size - 1)
        visible = d <= dmap[py, px] + depth_tol
        # bilinear sample of the prediction
        grid = torch.stack([x / 0.5, -(y / 0.5)], -1)[None, None]
        val = F.grid_sample(m.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T
        wk = cos ** 2 * visible
        acc += val * wk[:, None]
        wsum += wk
        best = torch.maximum(best, cos * visible)
    out = torch.zeros(h, w, acc.shape[-1], device="cuda")
    out[mask] = acc / wsum.clamp(min=1e-8)[:, None]
    seen = torch.zeros(h, w, dtype=torch.bool, device="cuda")
    seen[mask] = wsum > 1e-6
    best_img = torch.zeros(h, w, device="cuda")
    best_img[mask] = best
    return out.cpu().numpy(), best_img.cpu().numpy(), seen.cpu().numpy()


class SuperMat:
    """Loads one SuperMat network at a time (multi-view, UV refinement bc, UV refinement rm)."""

    def __init__(self, sm_dir, ckpt_dir, base_model, seed):
        sys.path.insert(0, sm_dir)
        os.chdir(sm_dir)  # SuperMat imports its src/ package relative to the repo
        import inference_supermat_mv as mv
        import inference_uv_refine as uvr
        self.mv, self.uvr = mv, uvr
        self.ckpt_dir, self.seed = ckpt_dir, seed
        self.common = dict(base_model=base_model, device="cuda", seed=seed, disable_xformers=False)
        self.pipe, self.stage = None, None

    def load(self, stage):
        self.pipe = None
        torch.cuda.empty_cache()
        if stage == "mv":
            a = argparse.Namespace(checkpoint=f"{self.ckpt_dir}/supermat_mv.pth", image_size=512, num_views=6,
                                   use_camera_embeds=True, save_orm=False, **self.common)
            self.pipe = self.mv.build_pipeline(a)
        else:
            a = argparse.Namespace(checkpoint=f"{self.ckpt_dir}/uv_refine_{stage}.pth", image_size=1024, **self.common)
            self.pipe = self.uvr.build_pipeline(a)
        self.stage = stage

    def multiview(self, case_dir, out_dir):
        self.mv.run_case(pipe=self.pipe, input_dir=Path(case_dir), output_dir=Path(out_dir), image_size=512,
                         num_views=6, use_camera_embeds=True, seed=self.seed, save_orm=False,
                         device=torch.device("cuda"))

    def refine_uv(self, uv_path, pos_path, mask_path, size=1024):
        from src.utils import load_rgb_tensor, load_mask_tensor, to_uint8_rgb
        if self.seed is not None:
            torch.manual_seed(self.seed)
        dev = torch.device("cuda")
        out = self.uvr.run_uv_refine(self.pipe, load_rgb_tensor(Path(uv_path), size, dev),
                                     load_rgb_tensor(Path(pos_path), size, dev), load_mask_tensor(Path(mask_path), size, dev))
        return to_uint8_rgb(out[0])


def stage_project(sm, path, out_dir, args):
    """Renders, multi-view prediction and back-projection; writes tex<k>_*_proj.png, position map and mask."""
    t0 = time.time()
    parts = load_parts(path)
    center, scale = normalization(np.concatenate([p["mesh"].vertices for p in parts]))
    rot = lambda n: np.stack([n[:, 0], -n[:, 2], n[:, 1]], 1).astype(np.float32)  # same axis swap as normalize
    pv = []
    for p in parts:
        v = normalize(np.asarray(p["mesh"].vertices), center, scale)
        pv.append({"v": v * EXTENT, "f": np.asarray(p["mesh"].faces, np.int64), "uv": p["uv"], "texture": p["texture"],
                   "n": rot(np.asarray(p["mesh"].vertex_normals)), "v_unit": v})
    case = out_dir / "supermat_views"
    case.mkdir(exist_ok=True)
    images, depths = render_views(pv, args.render_size)
    for k, im in enumerate(images):
        Image.fromarray(to_u8(im.cpu().numpy()), "RGBA").save(case / f"color_{k:04d}.png")
    (case / "meta.json").write_text(json.dumps({"locations": [
        {"index": f"{k:04d}", "transform_matrix": c2w_matrix(pos, up).tolist()} for k, (pos, up) in enumerate(VIEWS)]}))
    pred_dir = out_dir / "supermat_pred"
    sm.multiview(case, pred_dir)
    load = lambda n, k, mode: torch.tensor(np.asarray(Image.open(pred_dir / f"{n}_{k:04d}.png").convert(mode).resize(
        (args.render_size, args.render_size), Image.BILINEAR)), dtype=torch.float32, device="cuda") / 255
    albedo_v = [load("albedo", k, "RGB") for k in range(6)]
    rm_v = [torch.stack([load("roughness", k, "L"), load("metallic", k, "L")], -1) for k in range(6)]

    atlases = {}
    for k, p in enumerate(parts):
        atlases.setdefault(p["tex_key"], []).append(k)
    atlas_meta = []
    for a, (key, idx) in enumerate(atlases.items()):
        h, w = parts[idx[0]]["texture"].shape[:2]
        if h * w <= 1:
            h = w = args.texture_size
        bc = np.zeros((h, w, 3), np.float32); rmm = np.zeros((h, w, 2), np.float32)
        posmap = np.full((h, w, 3), 0.5, np.float32); cover = np.zeros((h, w), bool)
        seen = np.zeros((h, w), bool); best = np.zeros((h, w), np.float32)
        for k in idx:
            q = pv[k]
            mask, pos, nrm = texel_samples(q["uv"], q["f"], q["v"], q["n"], h, w)
            _, pos_unit, _ = texel_samples(q["uv"], q["f"], q["v_unit"], q["n"], h, w)
            vals_bc, b, sn = back_project(albedo_v, depths, pos, nrm, mask, args.render_size, args.depth_tol)
            vals_rm, _, _ = back_project(rm_v, depths, pos, nrm, mask, args.render_size, args.depth_tol)
            m = mask.cpu().numpy()
            bc[m], rmm[m], best[m], seen[m] = vals_bc[m], vals_rm[m], b[m], sn[m]
            posmap[m] = pos_unit.cpu().numpy()[m] + 0.5
            cover |= m
        # mask levels as in SuperMat's UV example: 0 outside, 1/3 unseen, 2/3 seen, 1 grazing views only
        mask_img = np.zeros((h, w), np.uint8)
        mask_img[cover & ~seen] = 85
        mask_img[cover & seen] = 170
        mask_img[cover & seen & (best < 0.35)] = 255
        known = cover & seen
        bc_u8 = dilate(to_u8(bc), known) if known.any() else to_u8(bc)
        rm_u8 = to_u8(np.concatenate([np.zeros((h, w, 1), np.float32), rmm], -1))
        rm_u8 = dilate(rm_u8, known) if known.any() else rm_u8
        Image.fromarray(bc_u8).save(out_dir / f"tex{a}_basecolor_proj.png")
        Image.fromarray(rm_u8).save(out_dir / f"tex{a}_rm_proj.png")
        Image.fromarray(to_u8(posmap)).save(out_dir / f"tex{a}_uvpos.png")
        Image.fromarray(mask_img).save(out_dir / f"tex{a}_supermat_mask.png")
        Image.fromarray(to_u8(cover.astype(np.float32))).save(out_dir / f"tex{a}_mask.png")
        atlas_meta.append({"index": a, "size": [h, w], "parts": [parts[k]["name"] for k in idx],
                           "coverage": round(float(cover.mean()), 4),
                           "seen_fraction": round(float(known.sum() / max(cover.sum(), 1)), 4)})
    (out_dir / "stage.json").write_text(json.dumps({"atlases": atlas_meta, "time_project_s": round(time.time() - t0, 1)}))


def stage_refine(sm, out_dir, key, args):
    st = json.loads((out_dir / "stage.json").read_text())
    for at in st["atlases"]:
        a, (h, w) = at["index"], at["size"]
        src = out_dir / (f"tex{a}_basecolor_proj.png" if key == "bc" else f"tex{a}_rm_proj.png")
        if args.no_refine:
            img = np.asarray(Image.open(src).convert("RGB"))
        else:
            img = sm.refine_uv(src, out_dir / f"tex{a}_uvpos.png", out_dir / f"tex{a}_supermat_mask.png")
            img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
        Image.fromarray(img).save(out_dir / (f"tex{a}_basecolor.png" if key == "bc" else f"tex{a}_rm.png"))


def stage_done(out_dir, stage):
    """True if this stage's outputs exist (resume after an interruption)."""
    st = out_dir / "stage.json"
    if not st.exists():
        return False
    if stage == "project":
        return True
    suffix = "basecolor" if stage == "bc" else "rm"
    return all((out_dir / f"tex{a['index']}_{suffix}.png").exists() for a in json.loads(st.read_text())["atlases"])


def finalize(path, out_dir, args, t_total):
    parts = load_parts(path)
    st = json.loads((out_dir / "stage.json").read_text())
    by_name = {}
    for at in st["atlases"]:
        a = at["index"]
        cover = np.asarray(Image.open(out_dir / f"tex{a}_mask.png")) > 127
        bc = np.asarray(Image.open(out_dir / f"tex{a}_basecolor.png").convert("RGB"))
        rm = np.asarray(Image.open(out_dir / f"tex{a}_rm.png").convert("RGB"))
        rough, metal = rm[..., 1], rm[..., 2]
        Image.fromarray(rough).save(out_dir / f"tex{a}_roughness.png")
        Image.fromarray(metal).save(out_dir / f"tex{a}_metallic.png")
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(dilate(bc, cover)),
            metallicRoughnessTexture=Image.fromarray(np.stack(
                [np.zeros_like(rough), dilate(rough, cover), dilate(metal, cover)], -1)),
            metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)
        for n in at["parts"]:
            by_name[n] = mat
    glb = []
    for p in parts:
        mesh = p["mesh"].copy()
        mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=by_name[p["name"]])
        glb.append(mesh)
    trimesh.Scene(glb).export(out_dir / "textured.glb")
    meta = {"input": str(path), "method": "supermat", "parts": [p["name"] for p in parts], "atlases": st["atlases"],
            "seed": args.seed, "refine": not args.no_refine, "time_project_s": st["time_project_s"],
            "total_s": round(t_total, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--pattern", default="**/*_baked.glb")
    ap.add_argument("--out", default="results/supermat")
    ap.add_argument("--sm-dir", required=True, help="clone of https://github.com/hyj542682306/SuperMat")
    ap.add_argument("--ckpt-dir", required=True, help="SuperMat checkpoints (oyiya/SuperMat on Hugging Face)")
    ap.add_argument("--base-model", default="sd2-community/stable-diffusion-2-1")
    ap.add_argument("--render-size", type=int, default=1024)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depth-tol", type=float, default=0.01, help="visibility test tolerance (object size 0.9)")
    ap.add_argument("--no-refine", action="store_true", help="skip SuperMat's UV refinement")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out_root = Path(args.out).resolve()
    assets = [inp] if inp.is_file() else sorted(inp.glob(args.pattern))
    if not assets:
        sys.exit(f"no assets found in {inp} with pattern {args.pattern}")
    sm = SuperMat(args.sm_dir, args.ckpt_dir, args.base_model, args.seed)
    todo = []
    for path in assets:
        name = path.stem if inp.is_file() else str(path.relative_to(inp).with_suffix("")).replace(os.sep, "__")
        out_dir = out_root / name
        if (out_dir / "meta.json").exists() and not args.overwrite:
            print(f"skip {name} (done)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error.txt").unlink(missing_ok=True)
        todo.append((name, path, out_dir))

    t0, ok = time.time(), []
    stages = [("project", "mv")] + ([] if args.no_refine else [("bc", "bc"), ("rm", "rm")])
    for stage, net in stages:
        sm.load(net)
        for i, (name, path, out_dir) in enumerate(todo):
            if (out_dir / "error.txt").exists():
                continue
            if stage_done(out_dir, stage):
                continue
            try:
                if stage == "project":
                    stage_project(sm, path, out_dir, args)
                else:
                    stage_refine(sm, out_dir, stage, args)
                print(f"[{stage} {i + 1}/{len(todo)}] {name}", flush=True)
            except Exception as e:
                print(f"[{stage} {i + 1}/{len(todo)}] FAILED {name}: {type(e).__name__}: {e}", flush=True)
                (out_dir / "error.txt").write_text(traceback.format_exc())
    if args.no_refine:
        for name, path, out_dir in todo:
            if not (out_dir / "error.txt").exists():
                stage_refine(sm, out_dir, "bc", args); stage_refine(sm, out_dir, "rm", args)
    per_asset = (time.time() - t0) / max(len(todo), 1)
    for name, path, out_dir in todo:
        if (out_dir / "error.txt").exists():
            continue
        try:
            finalize(path, out_dir, args, per_asset)
        except Exception as e:
            print(f"FAILED finalize {name}: {type(e).__name__}: {e}", flush=True)
            (out_dir / "error.txt").write_text(traceback.format_exc())
    print(f"done: {len(todo)} assets, {per_asset:.1f} s per asset", flush=True)


if __name__ == "__main__":
    main()
