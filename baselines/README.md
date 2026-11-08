# Baselines and evaluation

| File | Purpose |
|---|---|
| `unguided_trellis2.py` | Unguided TRELLIS.2 baseline. Runs the material stage conditioned on the geometry and one render of the baked asset, then bakes the voxel materials into the original UV layout. |
| `evaluate_uv.py` | Per-channel MAE / PSNR (base color, roughness, metallic) in UV space against `<base>_pbr.glb`, on the texels covered by the UV layout. |
| `evaluate_relit.py` | Renders prediction and ground truth with the TRELLIS.2 PBR renderer under every HDR in `env_maps/` from 4 views; MAE / PSNR on object pixels, SSIM and LPIPS on the full image. Takes any GLB with the input geometry (`--pred-glb`), so it can be reused for other methods. |
| `summarize.py` | Mean ± std of the metrics over several runs (e.g. seeds). |
| `run_unguided.sh` | Baseline and both evaluations on a dataset folder. |

Both evaluations also report an input-as-albedo reference (baked texture as base color, roughness 0.5, metallic 0).

## Usage

```sh
export ATTN_BACKEND=xformers
bash baselines/run_unguided.sh pbr_dataset/demo 1024     # <base>_baked.glb next to <base>_pbr.glb
```

Single asset with a given condition image instead of the default render:

```sh
python baselines/unguided_trellis2.py --input examples/cauldron_baked.glb --out results/cauldron \
    --cond-image examples/caudron_render.png
python baselines/evaluate_uv.py --pred results/cauldron
python baselines/evaluate_relit.py --pred results/cauldron
```

Output per asset: `cond.png`, `tex<k>_{basecolor,roughness,metallic,mask}.png` (one set per input
texture atlas, same size as the input texture), `textured.glb`, `meta.json`, and `relit/`
(ground truth | prediction | input-as-albedo renders).

The default condition image is an unlit render of the baked texture from the glTF front (+Z) at
15° elevation, with alpha, so background removal is never applied (`--view-yaw`, `--view-pitch`).

## Notes

- Resolution 1024 is the default. At 512 the metallic channel collapses to 1 everywhere for some
  seeds (2 of 3 on the cauldron, also with the unmodified `Trellis2TexturingPipeline.run`).
- UVs outside [0, 1] (glTF repeat wrapping) are handled by shifting each triangle into the unit tile.
- Normal maps are not used in the relighting evaluation (face normals for all methods).

## Environment

- `transformers` 4.x (tested 4.57.1). 5.x renamed DINOv3's layers.
- `opencv-python-headless` 4.x (tested 4.11.0.86). The 5.0 wheel cannot read `.exr`, which the
  environment maps need.
- xformers 0.0.31 on Blackwell (sm_120): it dispatches to its FlashAttention-3 kernel, which is
  Hopper-only and crashes. Set `_USE_FLASH_ATTENTION_3 = False` in `xformers/ops/fmha/dispatch.py`.
- Building the CUDA extensions with `TORCH_CUDA_ARCH_LIST="8.0+PTX"` gives wheels that run on
  Ampere and (via PTX JIT) on Blackwell. CuMesh needs CCCL ≥ 2.7 headers when built with CUDA 12.0.
- Gated weights: `facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0`.

## Speed

Per asset after the first one (the first run after a GPU change spends about a minute on Triton autotuning):

| GPU | 512 | 1024 |
|---|---|---|
| A30 24 GB | ~22 s, 3.2 GB | ~55 s, 8.2 GB |
| RTX PRO 6000 Blackwell (MIG 1g.24gb) | ~30 s, 3.1 GB | ~50–60 s, 6.9 GB |
