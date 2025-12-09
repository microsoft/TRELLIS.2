from .. import config
import importlib
import torch
import torch.nn as nn
from .. import SparseTensor


_backends = {}


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if config.CONV not in _backends:
            _backends[config.CONV] = importlib.import_module(f'..conv_{config.CONV}', __name__)
        _backends[config.CONV].sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[config.CONV].sparse_conv3d_forward(self, x)


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if config.CONV not in _backends:
            _backends[config.CONV] = importlib.import_module(f'..conv_{config.CONV}', __name__)
        _backends[config.CONV].sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[config.CONV].sparse_inverse_conv3d_forward(self, x)
