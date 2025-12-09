import torch
import torch.nn as nn
from . import VarLenTensor

__all__ = [
    'SparseReLU',
    'SparseSiLU',
    'SparseGELU',
    'SparseActivation'
]


class SparseReLU(nn.ReLU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))
    

class SparseSiLU(nn.SiLU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseGELU(nn.GELU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(self.activation(input.feats))
    
