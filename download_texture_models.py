"""
Download all models required for Trellis2TexturingPipeline.

Models downloaded:
  microsoft/TRELLIS.2-4B
    - texturing_pipeline.json
    - ckpts/shape_enc_next_dc_f16c32_fp16        (~400 MB)
    - ckpts/tex_dec_next_dc_f16c32_fp16          (~400 MB)
    - ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16   (~2.6 GB)
    - ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16  (~2.6 GB)
  briaai/RMBG-2.0  (background removal)          (~1.0 GB)
  facebook/dinov3-vitl16-pretrain-lvd1689m        (~1.1 GB)

Total: ~8.1 GB
"""

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
RMBG_REPO = "briaai/RMBG-2.0"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_DIRECT_FILENAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINO_DIRECT_URL = (
    "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    "?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYTdwYzV6YzI3bDczbmJxcjNhZXAyYnhkIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzQ0MTgxMDl9fX1dfQ__"
    "&Signature=qIEnfQZHBmwdp38IDvUjV3wkZMiNX3v9-K-YXheXCFRo97GBhwNHG-PbJkb1Fw9PECXYcklow%7EiJ%7EV3Qh8TfytTe2t%7ElcNs63kCoxmRuBB8%7ElUXD5R%7E0nKX1HqqZQzxV%7ECXUcDh6wKWX5dcEeMG39bVjOAhADvItOLCPQU-l%7ECP9NM2Nb9sXi2Eyp21gU5PNAL11ziKY2U4dRuKvjDn9EhfGS1cvftO8AFGsIOEapmWbEZ9lpWgEGFWiWbt6ajy%7EnK6NoiycLatTTNF-hFCmSiD6xavV7MfgtXHmFi8SE7Vvrv%7EwAjGuA0ZFOQnUNSCbVzyQ8Rj-S8PA1OGp28svxA__"
    "&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=3857627804367069"
)

TRELLIS_TEXTURE_FILES = [
    "texturing_pipeline.json",
    "ckpts/shape_enc_next_dc_f16c32_fp16.json",
    "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors",
    "ckpts/tex_dec_next_dc_f16c32_fp16.json",
    "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
]


def download_trellis(local_dir: Path, token: str = None):
    print(f"\n[1/3] Downloading TRELLIS.2-4B texture models -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in TRELLIS_TEXTURE_FILES:
        dest = local_dir / filename
        if dest.exists():
            print(f"  skip  {filename}")
            continue
        print(f"  fetch {filename} ...")
        hf_hub_download(
            repo_id=TRELLIS_REPO,
            filename=filename,
            local_dir=str(local_dir),
            token=token,
        )
    print("  done.")


def download_rmbg(local_dir: Path, token: str = None):
    print(f"\n[2/3] Downloading RMBG-2.0 (background removal) -> {local_dir}")
    if any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin")):
        print("  skip  (already exists)")
        return
    snapshot_download(
        repo_id=RMBG_REPO,
        local_dir=str(local_dir),
        token=token,
        ignore_patterns=["*.md", "*.txt", "*.png", "*.jpg"],
    )
    print("  done.")


def download_dino(local_dir: Path, token: str = None):
    print(f"\n[3/3] Downloading DINOv3-L ({DINO_DIRECT_FILENAME}) -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / DINO_DIRECT_FILENAME
    if dest.exists():
        print(f"  skip  {DINO_DIRECT_FILENAME} (already exists)")
        return
    import urllib.request
    print(f"  fetch {DINO_DIRECT_FILENAME} ...")
    urllib.request.urlretrieve(DINO_DIRECT_URL, str(dest))
    print("  done.")


def main():
    parser = argparse.ArgumentParser(description="Download TRELLIS.2 texture pipeline models")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent / "models",
        help="Root directory to save models (default: ./models)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    trellis_dir = args.model_dir / "TRELLIS.2-4B"
    rmbg_dir    = args.model_dir / "RMBG-2.0"
    dino_dir    = args.model_dir / "dinov3-vitl16-pretrain-lvd1689m"

    print(f"Model root : {args.model_dir.resolve()}")
    print("Est. size  : ~8.1 GB")

    download_trellis(trellis_dir, args.token)
    download_rmbg(rmbg_dir, args.token)
    download_dino(dino_dir, args.token)

    print("\nAll models downloaded.")
    print(f"\nLoad the pipeline with:")
    print(f"  Trellis2TexturingPipeline.from_pretrained(r'{trellis_dir}', config_file='texturing_pipeline.json')")


if __name__ == "__main__":
    main()
