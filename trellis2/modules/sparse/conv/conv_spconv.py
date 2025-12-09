import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
import spconv.pytorch as spconv


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    algo = None
    if config.SPCONV_ALGO == 'native':
        algo = spconv.ConvAlgo.Native
    elif config.SPCONV_ALGO == 'implicit_gemm':
        algo = spconv.ConvAlgo.MaskImplicitGemm
    if stride == 1 and (padding is None):
        self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
    else:
        self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
    self.padding = padding  


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
    new_data = self.conv(x.data)
    new_shape = [x.shape[0], self.conv.out_channels]
    new_layout = None if spatial_changed else x.layout

    if spatial_changed and (x.shape[0] != 1):
        # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
        fwd = new_data.indices[:, 0].argsort()
        bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
        sorted_feats = new_data.features[fwd]
        sorted_coords = new_data.indices[fwd]
        unsorted_data = new_data
        new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )

    if spatial_changed and (x.shape[0] != 1):
        out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
        out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)

    return out


def sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    spatial_changed = any(s != 1 for s in self.stride)
    if spatial_changed:
        # recover the original spconv order
        data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
        bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
        data = data.replace_feature(x.feats[bwd])
    else:
        data = x.data

    new_data = self.conv(data)
    new_shape = [x.shape[0], self.conv.out_channels]
    new_layout = None if spatial_changed else x.layout
    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )
    return out
