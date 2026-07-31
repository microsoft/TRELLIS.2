"""
MLX pipeline factory — creates an upstream Trellis2ImageTo3DPipeline
with MLX-backed model adapters.

The upstream PT pipeline handles all orchestration, sampling, and mesh
extraction. MLX models are injected via thin adapters that convert
torch→mlx→torch at model boundaries.
"""
import os
import gc
import json
import time
import logging

from trellis2.model_revisions import (
    DINOV3_REVISION,
    RMBG_REVISION,
    revision_for_repo,
)

import mlx.core as mx

logger = logging.getLogger(__name__)

from . import load_safetensors, remap_flow_model_weights, remap_vae_decoder_weights
from .flow_models import MlxSparseStructureFlowModel, MlxSLatFlowModel
from .vae_decoders import MlxSparseUnetVaeDecoder, MlxFlexiDualGridVaeDecoder
from .structure_decoder import load_structure_decoder
from .dinov3 import load_dinov3_from_hf
from .adapters import (
    MlxFlowModelAdapter,
    MlxStructureDecoderAdapter,
    MlxFlexiDualGridAdapter,
    MlxTexVaeDecoderAdapter,
    MlxImageCondAdapter,
)


def _resolve_hf_path(
    rel_path: str,
    *,
    cache_dir: str = None,
    local_files_only: bool = False,
) -> str:
    """Resolve 'org/repo/path/to/file' to local HF cache path."""
    from huggingface_hub import hf_hub_download
    parts = rel_path.split('/')
    repo_id = f"{parts[0]}/{parts[1]}"
    file_base = '/'.join(parts[2:])
    hub_kwargs = {
        "revision": revision_for_repo(repo_id),
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    json_path = hf_hub_download(repo_id, f"{file_base}.json", **hub_kwargs)
    hf_hub_download(repo_id, f"{file_base}.safetensors", **hub_kwargs)
    return json_path.rsplit('.json', 1)[0]


def _resolve_model_path(
    weights_path: str,
    rel_path: str,
    *,
    cache_dir: str = None,
    local_files_only: bool = False,
) -> str:
    """Resolve model path — local first, then HF Hub."""
    full = os.path.join(weights_path, rel_path)
    if os.path.exists(f"{full}.json"):
        return full
    return _resolve_hf_path(
        rel_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def _load_mlx_flow_model(path: str, config: dict):
    """Load an MLX flow model from config + safetensors."""
    args = config['args']
    if config['name'] == 'SparseStructureFlowModel':
        model = MlxSparseStructureFlowModel(
            resolution=args['resolution'],
            in_channels=args['in_channels'],
            model_channels=args['model_channels'],
            cond_channels=args['cond_channels'],
            out_channels=args['out_channels'],
            num_blocks=args['num_blocks'],
            num_heads=args.get('num_heads', 12),
            mlp_ratio=args.get('mlp_ratio', 5.3334),
            pe_mode=args.get('pe_mode', 'rope'),
            share_mod=args.get('share_mod', True),
            qk_rms_norm=args.get('qk_rms_norm', True),
            qk_rms_norm_cross=args.get('qk_rms_norm_cross', True),
        )
        is_sparse = False
    elif config['name'] in ('SLatFlowModel', 'ElasticSLatFlowModel'):
        model = MlxSLatFlowModel(
            resolution=args['resolution'],
            in_channels=args['in_channels'],
            model_channels=args['model_channels'],
            cond_channels=args['cond_channels'],
            out_channels=args['out_channels'],
            num_blocks=args['num_blocks'],
            num_heads=args.get('num_heads', 12),
            mlp_ratio=args.get('mlp_ratio', 5.3334),
            pe_mode=args.get('pe_mode', 'rope'),
            share_mod=args.get('share_mod', True),
            qk_rms_norm=args.get('qk_rms_norm', True),
            qk_rms_norm_cross=args.get('qk_rms_norm_cross', True),
        )
        is_sparse = True
    else:
        raise ValueError(f"Unknown flow model type: {config['name']}")

    weights = load_safetensors(f"{path}.safetensors")
    weights = remap_flow_model_weights(weights)
    model.load_weights(list(weights.items()))
    return MlxFlowModelAdapter(model, is_sparse=is_sparse)


def _load_mlx_structure_decoder(path: str, config: dict):
    """Load MLX structure decoder and wrap in adapter."""
    model = load_structure_decoder(path)
    return MlxStructureDecoderAdapter(model)


def _load_mlx_shape_decoder(path: str, config: dict):
    """Load MLX FlexiDualGrid shape decoder and wrap in adapter."""
    args = config['args']
    model = MlxFlexiDualGridVaeDecoder(
        resolution=args['resolution'],
        model_channels=args['model_channels'],
        latent_channels=args['latent_channels'],
        num_blocks=args['num_blocks'],
        block_type=args['block_type'],
        up_block_type=args['up_block_type'],
        block_args=args.get('block_args'),
        use_fp16=args.get('use_fp16', False),
    )
    weights = load_safetensors(f"{path}.safetensors")
    weights = remap_vae_decoder_weights(weights)
    weights = {f'decoder.{k}': v for k, v in weights.items()}
    model.load_weights(list(weights.items()))
    return MlxFlexiDualGridAdapter(model)


def _load_mlx_tex_decoder(path: str, config: dict):
    """Load MLX texture VAE decoder and wrap in adapter."""
    args = config['args']
    model = MlxSparseUnetVaeDecoder(
        out_channels=args['out_channels'],
        model_channels=args['model_channels'],
        latent_channels=args['latent_channels'],
        num_blocks=args['num_blocks'],
        block_type=args['block_type'],
        up_block_type=args['up_block_type'],
        block_args=args.get('block_args'),
        use_fp16=args.get('use_fp16', False),
        pred_subdiv=args.get('pred_subdiv', True),
    )
    weights = load_safetensors(f"{path}.safetensors")
    weights = remap_vae_decoder_weights(weights)
    model.load_weights(list(weights.items()))
    return MlxTexVaeDecoderAdapter(model)


# Map model name patterns to loader functions
_LOADER_MAP = {
    'sparse_structure_flow_model': _load_mlx_flow_model,
    'sparse_structure_decoder': _load_mlx_structure_decoder,
    'shape_slat_decoder': _load_mlx_shape_decoder,
    'tex_slat_decoder': _load_mlx_tex_decoder,
}


def _get_loader(name: str, config: dict):
    """Pick the right loader for a model name."""
    # Exact match first
    if name in _LOADER_MAP:
        return _LOADER_MAP[name]
    # Flow models by name pattern
    if 'flow_model' in name:
        return _load_mlx_flow_model
    # VAE decoders by config name
    if config['name'] == 'FlexiDualGridVaeDecoder':
        return _load_mlx_shape_decoder
    if config['name'] == 'SparseUnetVaeDecoder':
        return _load_mlx_tex_decoder
    raise ValueError(f"No loader for model '{name}' (type: {config['name']})")


def create_mlx_pipeline(
    weights_path: str = "weights/TRELLIS.2-4B",
    *,
    cache_dir: str = None,
    local_files_only: bool = False,
):
    """Create upstream Trellis2ImageTo3DPipeline with MLX-backed models.

    All model compute runs in MLX. The upstream PT pipeline handles
    orchestration, sampling (FlowEulerCfgSampler etc.), and mesh extraction.
    """
    import torch
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import samplers
    from trellis2.pipelines.rembg import BiRefNet

    print(f"[MLX] Loading pipeline config from {weights_path}...")
    config_file = os.path.join(weights_path, "pipeline.json")
    with open(config_file) as f:
        args = json.load(f)['args']

    # Load all models with MLX adapters
    models = {}
    for name, rel_path in args['models'].items():
        path = _resolve_model_path(
            weights_path,
            rel_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        with open(f"{path}.json") as f:
            model_config = json.load(f)

        t0 = time.time()
        loader = _get_loader(name, model_config)
        models[name] = loader(path, model_config)
        dt = time.time() - t0
        print(f"  [MLX] Loaded '{name}' in {dt:.1f}s")

    # Create upstream pipeline with MLX models
    pipeline = Trellis2ImageTo3DPipeline(models)
    pipeline._pretrained_args = args

    # Set up samplers (upstream PT classes — source of truth for sampling)
    pipeline.sparse_structure_sampler = getattr(
        samplers, args['sparse_structure_sampler']['name']
    )(**args['sparse_structure_sampler']['args'])
    pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

    pipeline.shape_slat_sampler = getattr(
        samplers, args['shape_slat_sampler']['name']
    )(**args['shape_slat_sampler']['args'])
    pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

    pipeline.tex_slat_sampler = getattr(
        samplers, args['tex_slat_sampler']['name']
    )(**args['tex_slat_sampler']['args'])
    pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

    # Normalization
    pipeline.shape_slat_normalization = args['shape_slat_normalization']
    pipeline.tex_slat_normalization = args['tex_slat_normalization']

    # Image conditioning (MLX DINOv3)
    pipeline.image_cond_model = MlxImageCondAdapter(
        load_dinov3_from_hf(
            args['image_cond_model']['args']['model_name'],
            revision=DINOV3_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )

    # Background removal (PT — lightweight, used once)
    pipeline.rembg_model = BiRefNet(
        **args['rembg_model']['args'],
        revision=RMBG_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    pipeline.low_vram = True
    pipeline._device = torch.device('cpu')
    pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
    pipeline.pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }

    print("[MLX] Pipeline ready.")
    return pipeline


def to_glb(mesh, output_path: str,
           decimation_target: int = 1000000,
           texture_size: int = 2048,
           remesh: bool = False,
           verbose: bool = True) -> str:
    """Export MeshWithVoxel to GLB file."""
    import o_voxel

    print(f"Exporting to {output_path}...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        verbose=verbose,
    )
    glb.export(output_path)
    print(f"Exported: {output_path}")
    return output_path


# Backward-compat alias
class MlxTrellis2Pipeline:
    """Deprecated — use create_mlx_pipeline() instead.

    Thin wrapper that creates the upstream pipeline and delegates .run()/.to_glb().
    """

    def __init__(self, weights_path: str = "weights/TRELLIS.2-4B"):
        self._pipeline = create_mlx_pipeline(weights_path)

    def run(self, image, **kwargs):
        return self._pipeline.run(image, **kwargs)

    def to_glb(self, mesh, output_path, **kwargs):
        return to_glb(mesh, output_path, **kwargs)
