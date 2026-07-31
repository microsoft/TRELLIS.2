"""
Thin adapters wrapping MLX models to match upstream PyTorch model interfaces.
Used by create_mlx_pipeline() so the upstream Trellis2ImageTo3DPipeline can
call MLX models transparently via torch→numpy→mx→numpy→torch conversion.
"""
import numpy as np
import torch
import torch.nn as nn
import mlx.core as mx

from .sparse_tensor import MlxSparseTensor
from trellis2.modules.sparse import SparseTensor


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _sparse_to_mlx(st: SparseTensor) -> MlxSparseTensor:
    """Convert upstream SparseTensor to MlxSparseTensor."""
    feats = mx.array(st.feats.cpu().numpy())
    coords = mx.array(st.coords.cpu().numpy().astype(np.int32))
    return MlxSparseTensor(feats=feats, coords=coords)


def _mlx_to_sparse(mx_st: MlxSparseTensor, ref: SparseTensor = None) -> SparseTensor:
    """Convert MlxSparseTensor back to upstream SparseTensor.

    If ref is provided and coords match, reuses backend data via .replace().
    """
    mx.eval(mx_st.feats)
    feats = torch.from_numpy(np.array(mx_st.feats))
    if ref is not None and mx_st.coords.shape == ref.coords.shape:
        return ref.replace(feats=feats)
    mx.eval(mx_st.coords)
    coords = torch.from_numpy(np.array(mx_st.coords)).int()
    return SparseTensor(feats=feats, coords=coords)


def _torch_to_mx(t: torch.Tensor) -> mx.array:
    """Convert a PyTorch CPU tensor to MLX array."""
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().cpu().numpy()).astype(mx.bfloat16)
    return mx.array(t.cpu().numpy())


def _mx_to_torch(a: mx.array) -> torch.Tensor:
    """Convert MLX array to PyTorch CPU tensor."""
    mx.eval(a)
    return torch.from_numpy(np.array(a))


# ---------------------------------------------------------------------------
# MlxFlowModelAdapter
# ---------------------------------------------------------------------------

class MlxFlowModelAdapter(nn.Module):
    """Wraps dense MlxSparseStructureFlowModel or sparse MlxSLatFlowModel.

    The upstream PT sampler calls model(x_t, t, cond, **kwargs).
    This adapter converts tensors, runs the MLX model, and converts back.
    """

    def __init__(self, mlx_model, is_sparse: bool = False):
        super().__init__()
        self._mlx = mlx_model
        self._is_sparse = is_sparse
        self.resolution = mlx_model.resolution
        self.in_channels = mlx_model.in_channels
        self.out_channels = mlx_model.out_channels

    def forward(self, x, t, cond, **kwargs):
        if self._is_sparse:
            return self._forward_sparse(x, t, cond, **kwargs)
        return self._forward_dense(x, t, cond)

    def _forward_dense(self, x, t, cond):
        mx_out = self._mlx(_torch_to_mx(x), _torch_to_mx(t), _torch_to_mx(cond))
        return _mx_to_torch(mx_out)

    def _forward_sparse(self, x, t, cond, **kwargs):
        mx_kwargs = {}
        if kwargs.get('concat_cond') is not None:
            mx_kwargs['concat_cond'] = _sparse_to_mlx(kwargs['concat_cond'])
        mx_out = self._mlx(
            _sparse_to_mlx(x), _torch_to_mx(t), _torch_to_mx(cond), **mx_kwargs
        )
        return _mlx_to_sparse(mx_out, x)

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def eval(self):
        return self


# ---------------------------------------------------------------------------
# MlxStructureDecoderAdapter
# ---------------------------------------------------------------------------

class MlxStructureDecoderAdapter(nn.Module):
    """Wraps MlxSparseStructureDecoder for the upstream pipeline."""

    def __init__(self, mlx_model):
        super().__init__()
        self._mlx = mlx_model

    def forward(self, z_s):
        return _mx_to_torch(self._mlx(_torch_to_mx(z_s)))

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def eval(self):
        return self


# ---------------------------------------------------------------------------
# MlxFlexiDualGridAdapter
# ---------------------------------------------------------------------------

class MlxFlexiDualGridAdapter(nn.Module):
    """Wraps MlxFlexiDualGridVaeDecoder to match upstream FlexiDualGridVaeDecoder.

    Upstream returns (meshes, subs) in eval mode with return_subs=True,
    where meshes is a list of Mesh objects. This adapter runs the MLX decoder,
    converts outputs to torch, then does mesh extraction via o_voxel.
    """

    def __init__(self, mlx_model):
        super().__init__()
        self._mlx = mlx_model
        self.low_vram = False  # upstream toggles this

    @property
    def resolution(self):
        return self._mlx.resolution

    def set_resolution(self, resolution):
        self._mlx.set_resolution(resolution)

    def forward(self, x, return_subs=False, **kwargs):
        from trellis2.representations import Mesh
        from o_voxel.convert import flexible_dual_grid_to_mesh

        mx_x = _sparse_to_mlx(x)
        result = self._mlx(mx_x, return_subs=return_subs)

        if return_subs:
            (h_mx, verts_mx, inter_mx, quad_mx), subs_mx = result
        else:
            (h_mx, verts_mx, inter_mx, quad_mx) = result
            subs_mx = None

        mx.eval(h_mx.feats, h_mx.coords, verts_mx, inter_mx, quad_mx)

        # Convert to torch for mesh extraction
        coords_t = torch.from_numpy(np.array(h_mx.coords[:, 1:])).int()
        verts_t = torch.from_numpy(np.array(verts_mx)).float()
        inter_t = torch.from_numpy(np.array(inter_mx)).bool()
        quad_t = torch.from_numpy(np.array(quad_mx)).float()

        mesh_verts, mesh_faces = flexible_dual_grid_to_mesh(
            coords_t, verts_t, inter_t, quad_t,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=self._mlx.resolution,
            train=False,
        )
        meshes = [Mesh(mesh_verts, mesh_faces)]

        if return_subs:
            subs_t = []
            for s in subs_mx:
                mx.eval(s.feats)
                subs_t.append(_mlx_to_sparse(s))
            return meshes, subs_t

        return meshes

    def upsample(self, x, upsample_times):
        mx_x = _sparse_to_mlx(x)
        mx_coords = self._mlx.upsample(mx_x, upsample_times)
        mx.eval(mx_coords)
        return torch.from_numpy(np.array(mx_coords)).int()

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def eval(self):
        return self


# ---------------------------------------------------------------------------
# MlxTexVaeDecoderAdapter
# ---------------------------------------------------------------------------

class MlxTexVaeDecoderAdapter(nn.Module):
    """Wraps MlxSparseUnetVaeDecoder (texture) for the upstream pipeline.

    Returns SparseTensor so upstream arithmetic (* 0.5 + 0.5) works.
    """

    def __init__(self, mlx_model):
        super().__init__()
        self._mlx = mlx_model

    def forward(self, x, guide_subs=None, **kwargs):
        mx_x = _sparse_to_mlx(x)
        mx_guide = None
        if guide_subs is not None:
            mx_guide = [_sparse_to_mlx(s) for s in guide_subs]

        result = self._mlx(mx_x, guide_subs=mx_guide)
        mx.eval(result.feats)
        return _mlx_to_sparse(result, x)

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def eval(self):
        return self


# ---------------------------------------------------------------------------
# MlxImageCondAdapter
# ---------------------------------------------------------------------------

class MlxImageCondAdapter:
    """Wraps MlxDINOv3FeatureExtractor for the upstream pipeline.

    The upstream pipeline sets .image_size then calls model(images).
    """

    def __init__(self, mlx_dino):
        self._mlx = mlx_dino
        self.image_size = 512

    def __call__(self, images):
        from PIL import Image as PILImage
        resized = [
            img.resize((self.image_size, self.image_size), PILImage.LANCZOS)
            for img in images
        ]
        mx_out = self._mlx(resized)
        mx.eval(mx_out)
        return torch.from_numpy(np.array(mx_out))

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self
