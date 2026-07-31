"""Pinned Hugging Face inputs used by the supported inference CLIs."""

TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"

DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"

RMBG_REPO = "briaai/RMBG-2.0"
RMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"

TRELLIS_IMAGE_LARGE_REPO = "microsoft/TRELLIS-image-large"
TRELLIS_IMAGE_LARGE_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"

MODEL_REVISIONS = {
    TRELLIS_REPO: TRELLIS_REVISION,
    TRELLIS_IMAGE_LARGE_REPO: TRELLIS_IMAGE_LARGE_REVISION,
    DINOV3_REPO: DINOV3_REVISION,
    RMBG_REPO: RMBG_REVISION,
}

# Exact runtime files for the supported image-to-3D CLI. Keeping this manifest
# avoids caching unrelated encoders, legacy checkpoints, and every ONNX/RMBG
# weight variant while still supporting 512, 1024, and 1024_cascade offline.
MODEL_FILES = {
    TRELLIS_REPO: (
        "pipeline.json",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.json",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.json",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    ),
    TRELLIS_IMAGE_LARGE_REPO: (
        "ckpts/ss_dec_conv3d_16l8_fp16.json",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    DINOV3_REPO: (
        "config.json",
        "model.safetensors",
    ),
    RMBG_REPO: (
        "config.json",
        "BiRefNet_config.py",
        "birefnet.py",
        "model.safetensors",
    ),
}

SOURCE_REVISIONS = {
    "pedronaugusto/mtlgemm": "867aec8234299a7fe1ede7f802c8debe5a939a82",
    "pedronaugusto/mtldiffrast": "4668cd91cb6d27f5e264731f94a06841fbf7aab8",
    "pedronaugusto/mtlbvh": "23f441c470ce1f537e1fd836f3ffb5b8245f7975",
    "pedronaugusto/mtlmesh": "212079e55772cff3d648a21372392c37e0643f3b",
    "EasternJournalist/utils3d": "9a4eb15e4021b67b12c460c7057d642626897ec8",
    "pedronaugusto/trellis2-apple": "6055b868734af6e12769d229d90580e775fae9f0",
    "shivampkumar/trellis-mac": "d58628f4f5b9c3de8274cb110074154f4b31cef2",
}


def revision_for_repo(repo_id, default=None):
    return MODEL_REVISIONS.get(repo_id, default)
