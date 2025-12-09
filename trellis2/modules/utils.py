import torch
import torch.nn as nn
from ..modules import sparse as sp

MIX_PRECISION_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    sp.SparseConv3d,
    sp.SparseInverseConv3d,
    sp.SparseLinear,
)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, MIX_PRECISION_MODULES):
        for p in l.parameters():
            p.data = p.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, MIX_PRECISION_MODULES):
        for p in l.parameters():
            p.data = p.data.float()


def convert_module_to(l, dtype):
    """
    Convert primitive modules to the given dtype.
    """
    if isinstance(l, MIX_PRECISION_MODULES):
        for p in l.parameters():
            p.data = p.data.to(dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def manual_cast(tensor, dtype):
    """
    Cast if autocast is not enabled.
    """
    if not torch.is_autocast_enabled():
        return tensor.type(dtype)
    return tensor


def str_to_dtype(dtype_str: str):
    return {
        'f16': torch.float16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'f32': torch.float32,
        'fp32': torch.float32,
        'float32': torch.float32,
    }[dtype_str]
