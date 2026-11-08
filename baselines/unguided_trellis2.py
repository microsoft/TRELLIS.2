"""Unguided TRELLIS.2 baseline: PBR materials for a mesh with baked-in lighting.

Runs the TRELLIS.2 material stage without guidance, conditioned on the geometry and one
render of the baked asset. The stock Trellis2TexturingPipeline drops the input UVs and
re-unwraps; here the per-voxel materials are baked into the mesh's original UV layout at
the resolution of its input texture, so the output maps are pixel-aligned with the
ground-truth PBR maps.

    python baselines/unguided_trellis2.py --input pbr_dataset/demo --out results/unguided_trellis2

Output per asset:
    cond.png                                    image condition
    tex<k>_{basecolor,roughness,metallic,mask}.png  one set per input texture atlas
    textured.glb                                input geometry with the predicted material
    meta.json                                   settings, timings, peak VRAM
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

import flex_gemm
import nvdiffrast.torch as dr


def load_parts(path):
    """Load an asset as a list of parts (scene transforms applied), each with its own
    UVs and baked base-color texture."""
    scene = trimesh.load(path, force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, gname = scene.graph[node]
        geom = scene.geometry[gname]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        geom = geom.copy()
        geom.apply_transform(transform)
        uv = getattr(geom.visual, "uv", None)
        if uv is None or len(uv) != len(geom.vertices):
            raise ValueError(f"{path}: part {gname} has no per-vertex UVs")
        tex = base_color_texture(geom.visual)
        parts.append({"name": gname, "mesh": geom, "uv": np.asarray(uv, dtype=np.float32), "texture": tex,
                      "tex_key": hashlib.md5(tex.tobytes()).hexdigest() + str(tex.shape)})
    if not parts:
        raise ValueError(f"{path}: no triangle meshes found")
    return parts


def base_color_texture(visual):
    """Baked texture as float RGB in [0, 1] (texture space, i.e. sRGB-encoded); a 1x1
    image if the material only has a constant color."""
    mat = getattr(visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    factor = getattr(mat, "baseColorFactor", None)
    factor = np.ones(3, dtype=np.float32) if factor is None else np.asarray(factor[:3], dtype=np.float32) / 255
    if img is None:
        return factor[None, None].astype(np.float32)
    return (np.asarray(img.convert("RGB"), dtype=np.float32) / 255 * factor).astype(np.float32)


def normalization(vertices):
    """Center and scale exactly like Trellis2TexturingPipeline.preprocess_mesh."""
    vmin, vmax = vertices.min(0), vertices.max(0)
    return (vmin + vmax) / 2, 0.99999 / (vmax - vmin).max()


def normalize(vertices, center, scale):
    """Into TRELLIS space: unit cube around the origin, Z up (glTF is Y up)."""
    v = (vertices - center) * scale
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1).astype(np.float32)


def look_at(eye, target=(0, 0, 0), up=(0, 0, 1)):
    eye, target, up = (np.asarray(x, dtype=np.float64) for x in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_deg, near=0.1, far=10.0):
    f = 1 / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]])


def render_condition(parts_norm, yaw, pitch, size=1024, fov=40.0):
    """Unlit render of the baked texture (RGBA, alpha = coverage). yaw = pitch = 0 looks
    at the glTF front (+Z), which is -Y in TRELLIS space."""
    yaw, pitch = np.radians(yaw), np.radians(pitch)
    dist = 0.9 / np.sin(np.radians(fov) / 2)  # the unit cube's bounding sphere fits
    eye = dist * np.array([np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    mvp = torch.tensor(perspective(fov) @ look_at(eye), dtype=torch.float32, device="cuda")

    verts, faces, uvs, face_part, offset = [], [], [], [], 0
    for k, p in enumerate(parts_norm):
        verts.append(p["v"]); uvs.append(p["uv"]); faces.append(p["f"] + offset)
        face_part.append(np.full(len(p["f"]), k)); offset += len(p["v"])
    v = torch.tensor(np.concatenate(verts), device="cuda")
    f = torch.tensor(np.concatenate(faces), dtype=torch.int32, device="cuda")
    uv = torch.tensor(np.concatenate(uvs), device="cuda")
    face_part = torch.tensor(np.concatenate(face_part), device="cuda")

    clip = (torch.cat([v, torch.ones_like(v[:, :1])], 1) @ mvp.T)[None]
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), clip, f, (size, size))
    tri = rast[0, ..., 3].long() - 1  # -1 = background
    uv_px = dr.interpolate(uv[None], rast, f)[0][0]
    rgb = torch.zeros(size, size, 3, device="cuda")
    for k, p in enumerate(parts_norm):
        m = (tri >= 0) & (face_part[tri.clamp(min=0)] == k)
        if not m.any():
            continue
        tex = torch.tensor(p["texture"], device="cuda").permute(2, 0, 1)[None]
        uv_m = uv_px[m] % 1  # repeat wrapping
        u, w = uv_m[:, 0], uv_m[:, 1]
        grid = torch.stack([u * 2 - 1, (1 - w) * 2 - 1], -1)[None, None]  # v=0 is the bottom row
        rgb[m] = F.grid_sample(tex, grid, mode="bilinear", align_corners=False)[0, :, 0].T
    rgba = torch.cat([rgb, (tri >= 0).float()[..., None]], -1).flip(0)  # GL rows are bottom-up
    return Image.fromarray((rgba.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), "RGBA")


@torch.no_grad()
def bake_to_uv(pbr_voxel, resolution, v_norm, faces, uv, height, width, layout):
    """Sample the per-voxel materials at every texel of the ORIGINAL UV layout (same as
    the stock postprocess_mesh, without re-unwrapping). resolution is the voxel grid
    size the shape was encoded at (spatial_shape is only the occupied bounding box)."""
    # UVs may lie in any integer tile (glTF REPEAT wrapping, e.g. v in [-1, 0]): shift
    # every triangle into [0, 1]^2 by the floor of its centroid. Triangles are
    # rasterized unindexed so that the shift can differ between triangles.
    uv_tri = torch.tensor(uv, device="cuda")[torch.tensor(faces, device="cuda")]  # [F, 3, 2]
    uv_tri = (uv_tri - torch.floor(uv_tri.mean(1, keepdim=True))).reshape(-1, 2)
    pos_tri = torch.tensor(v_norm, device="cuda")[torch.tensor(faces, device="cuda")].reshape(-1, 3)
    f = torch.arange(len(uv_tri), dtype=torch.int32, device="cuda").reshape(-1, 3)
    uv_clip = torch.cat([uv_tri[:, :1] * 2 - 1, (1 - uv_tri[:, 1:]) * 2 - 1,
                         torch.zeros_like(uv_tri[:, :1]), torch.ones_like(uv_tri[:, :1])], -1)[None]
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), uv_clip, f, (height, width))
    mask = rast[0, ..., 3] > 0  # row 0 = v=1 = top row of the texture image
    if not mask.any():
        raise ValueError("no texel is covered by the UV layout")
    pos = dr.interpolate(pos_tri[None], rast, f)[0][0]
    attrs = torch.zeros(height, width, pbr_voxel.shape[1], device="cuda")
    attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
        pbr_voxel.feats, pbr_voxel.coords,
        shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
        grid=((pos[mask] + 0.5) * resolution).reshape(1, -1, 3), mode="trilinear",
    )
    attrs = attrs.clamp(0, 1).cpu().numpy()
    return {"basecolor": attrs[..., layout["base_color"]],
            "metallic": attrs[..., layout["metallic"]][..., 0],
            "roughness": attrs[..., layout["roughness"]][..., 0],
            "mask": mask.cpu().numpy()}


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def dilate(img_u8, mask):
    """Fill texels outside the UV islands so that the GLB has no dark seams."""
    return cv2.inpaint(img_u8, (~mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)


def find_cond_image(path, args):
    """A given RGBA render to condition on, or None to render the baked asset ourselves."""
    if args.cond_image:
        return Path(args.cond_image)
    if args.cond_pattern:
        base = path.stem[:-len("_baked")] if path.stem.endswith("_baked") else path.stem
        hits = sorted(path.parent.glob(args.cond_pattern.format(base=base)))
        if hits:
            return hits[0]
    return None


def load_pipeline(model):
    from trellis2.pipelines import Trellis2TexturingPipeline

    pipe = Trellis2TexturingPipeline.from_pretrained(model, config_file="texturing_pipeline.json")
    pipe.cuda()
    return pipe


@torch.no_grad()
def run_asset(pipe, path, out_dir, args):
    t = {}
    t0 = time.time()
    parts = load_parts(path)
    all_v = np.concatenate([p["mesh"].vertices for p in parts])
    center, scale = normalization(all_v)
    parts_norm, offset, faces = [], 0, []
    for p in parts:
        v = normalize(np.asarray(p["mesh"].vertices), center, scale)
        f = np.asarray(p["mesh"].faces, dtype=np.int64)
        parts_norm.append({"v": v, "f": f, "uv": p["uv"], "texture": p["texture"]})
        faces.append(f + offset)
        offset += len(v)
    geometry = trimesh.Trimesh(np.concatenate([p["v"] for p in parts_norm]),
                               np.concatenate(faces), process=False)
    t["load"] = time.time() - t0

    t0 = time.time()
    cond_path = find_cond_image(path, args)
    if cond_path is None:
        cond_img = render_condition(parts_norm, args.view_yaw, args.view_pitch)
    else:
        cond_img = Image.open(cond_path)
        cond_img.load()
    cond_img.save(out_dir / "cond.png")
    image = pipe.preprocess_image(cond_img)  # has alpha, so only cropped (no RMBG)
    t["render"] = time.time() - t0

    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    t0 = time.time()
    cond = pipe.get_cond([image], 512 if args.resolution == 512 else 1024)
    t["cond"] = time.time() - t0
    t0 = time.time()
    shape_slat = pipe.encode_shape_slat(geometry, args.resolution)
    t["shape_encode"] = time.time() - t0
    t0 = time.time()
    flow = pipe.models[f"tex_slat_flow_model_{args.resolution}"]
    params = {"steps": args.steps} if args.steps else {}
    tex_slat = pipe.sample_tex_slat(cond, flow, shape_slat, params)
    t["sample"] = time.time() - t0
    t0 = time.time()
    pbr_voxel = pipe.decode_tex_slat(tex_slat)
    t["decode"] = time.time() - t0

    t0 = time.time()
    atlases = {}  # tex_key -> indices of the parts sharing that texture
    for k, p in enumerate(parts):
        atlases.setdefault(p["tex_key"], []).append(k)
    materials, atlas_meta = {}, []
    for a, (key, idx) in enumerate(atlases.items()):
        h, w = parts[idx[0]]["texture"].shape[:2]
        if h * w <= 1:  # constant input color: no texture to align with
            h = w = args.texture_size
        bc = np.zeros((h, w, 3), np.float32)
        rough = np.zeros((h, w), np.float32)
        metal = np.zeros((h, w), np.float32)
        mask = np.zeros((h, w), bool)
        for k in idx:
            pn = parts_norm[k]
            maps = bake_to_uv(pbr_voxel, args.resolution, pn["v"], pn["f"], pn["uv"], h, w, pipe.pbr_attr_layout)
            m = maps["mask"]
            bc[m], rough[m], metal[m] = maps["basecolor"][m], maps["roughness"][m], maps["metallic"][m]
            mask |= m
        bc, rough, metal = to_u8(bc), to_u8(rough), to_u8(metal)
        Image.fromarray(bc).save(out_dir / f"tex{a}_basecolor.png")
        Image.fromarray(rough).save(out_dir / f"tex{a}_roughness.png")
        Image.fromarray(metal).save(out_dir / f"tex{a}_metallic.png")
        Image.fromarray(to_u8(mask.astype(np.float32))).save(out_dir / f"tex{a}_mask.png")
        rough_d, metal_d = dilate(rough, mask), dilate(metal, mask)
        materials[key] = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(dilate(bc, mask)),
            metallicRoughnessTexture=Image.fromarray(
                np.stack([np.zeros_like(rough_d), rough_d, metal_d], -1)),
            metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True,
        )
        atlas_meta.append({"index": a, "size": [h, w], "parts": [parts[k]["name"] for k in idx],
                           "coverage": round(float(mask.mean()), 4)})
    glb_parts = []
    for p in parts:
        mesh = p["mesh"].copy()
        mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=materials[p["tex_key"]])
        glb_parts.append(mesh)
    trimesh.Scene(glb_parts).export(out_dir / "textured.glb")
    t["bake"] = time.time() - t0

    meta = {"input": str(path), "parts": [p["name"] for p in parts], "atlases": atlas_meta,
            "cond_image": str(cond_path) if cond_path else "render",
            "resolution": args.resolution, "seed": args.seed, "steps": args.steps,
            "view_yaw": args.view_yaw, "view_pitch": args.view_pitch,
            "n_voxels": int(shape_slat.coords.shape[0]),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "time_s": {k: round(v, 2) for k, v in t.items()},
            "total_s": round(sum(t.values()), 2), "gpu": torch.cuda.get_device_name()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="an asset file, or a directory searched with --pattern")
    ap.add_argument("--pattern", default="**/*_baked.glb", help="glob for the baked assets inside --input")
    ap.add_argument("--out", default="results/unguided_trellis2")
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--resolution", type=int, default=1024, choices=[512, 1024],
                    help="512 is faster but its metallic channel often collapses to 1 everywhere")
    ap.add_argument("--steps", type=int, default=None, help="sampler steps (default: pipeline config, 12)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--view-yaw", type=float, default=0.0, help="degrees around the up axis, 0 = glTF front")
    ap.add_argument("--view-pitch", type=float, default=15.0, help="degrees of elevation")
    ap.add_argument("--cond-image", default=None, help="RGBA image to condition on (single --input file only)")
    ap.add_argument("--cond-pattern", default=None,
                    help="glob next to each asset for its condition image, {base} = file name without "
                         "'_baked', e.g. '{base}_render.png'; assets without a match are rendered")
    ap.add_argument("--texture-size", type=int, default=1024, help="for parts without an input texture")
    ap.add_argument("--overwrite", action="store_true", help="redo assets that already have meta.json")
    args = ap.parse_args()

    inp = Path(args.input)
    if args.cond_image and not inp.is_file():
        sys.exit("--cond-image needs a single --input file, use --cond-pattern for a directory")
    assets = [inp] if inp.is_file() else sorted(inp.glob(args.pattern))
    if not assets:
        sys.exit(f"no assets found in {inp} with pattern {args.pattern}")
    out_root = Path(args.out)
    pipe = load_pipeline(args.model)

    for i, path in enumerate(assets):
        name = path.stem if inp.is_file() else str(path.relative_to(inp).with_suffix("")).replace(os.sep, "__")
        out_dir = out_root / name
        if (out_dir / "meta.json").exists() and not args.overwrite:
            print(f"[{i + 1}/{len(assets)}] skip {name} (done)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error.txt").unlink(missing_ok=True)
        try:
            meta = run_asset(pipe, path, out_dir, args)
            print(f"[{i + 1}/{len(assets)}] {name}: {meta['total_s']} s, {meta['n_voxels']} voxels, "
                  f"{meta['peak_vram_gb']} GB peak", flush=True)
        except Exception as e:  # log and continue with the next asset
            print(f"[{i + 1}/{len(assets)}] FAILED {name}: {type(e).__name__}: {e}", flush=True)
            (out_dir / "error.txt").write_text(traceback.format_exc())
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
