from typing import *
import torch
from . import _C


@torch.no_grad()
def encode_seq(coords: torch.Tensor, permute: List[int] = [0, 1, 2], mode: Literal['z_order', 'hilbert'] = 'z_order') -> torch.Tensor:
    """
    Encodes 3D coordinates into a 30-bit code.

    Args:
        coords: a tensor of shape [N, 3] containing the 3D coordinates.
        permute: the permutation of the coordinates.
        mode: the encoding mode to use.
    """
    assert coords.shape[-1] == 3 and coords.ndim == 2, "Input coordinates must be of shape [N, 3]"
    x = coords[:, permute[0]].int()
    y = coords[:, permute[1]].int()
    z = coords[:, permute[2]].int()
    if mode == 'z_order':
        if coords.device.type == 'cpu':
            return _C.z_order_encode_cpu(x, y, z)
        elif coords.device.type == 'cuda':
            return _C.z_order_encode_cuda(x, y, z)
        else:
            raise ValueError(f"Unsupported device type: {coords.device.type}")
    elif mode == 'hilbert':
        if coords.device.type == 'cpu':
            return _C.hilbert_encode_cpu(x, y, z)
        elif coords.device.type == 'cuda':
            return _C.hilbert_encode_cuda(x, y, z)
        else:
            raise ValueError(f"Unsupported device type: {coords.device.type}")
    else:
        raise ValueError(f"Unknown encoding mode: {mode}")


@torch.no_grad()
def decode_seq(code: torch.Tensor, permute: List[int] = [0, 1, 2], mode: Literal['z_order', 'hilbert'] = 'z_order') -> torch.Tensor:
    """
    Decodes a 30-bit code into 3D coordinates.

    Args:
        code: a tensor of shape [N] containing the 30-bit code.
        permute: the permutation of the coordinates.
        mode: the decoding mode to use.
    """
    assert code.ndim == 1, "Input code must be of shape [N]"
    if mode == 'z_order':
        if code.device.type == 'cpu':
            coords = _C.z_order_decode_cpu(code)
        elif code.device.type == 'cuda':
            coords = _C.z_order_decode_cuda(code)
        else:
            raise ValueError(f"Unsupported device type: {code.device.type}")
    elif mode == 'hilbert':
        if code.device.type == 'cpu':
            coords = _C.hilbert_decode_cpu(code)
        elif code.device.type == 'cuda':
            coords = _C.hilbert_decode_cuda(code)
        else:
            raise ValueError(f"Unsupported device type: {code.device.type}")
    else:
        raise ValueError(f"Unknown decoding mode: {mode}")
    x = coords[permute.index(0)]
    y = coords[permute.index(1)]
    z = coords[permute.index(2)]
    return torch.stack([x, y, z], dim=-1)
