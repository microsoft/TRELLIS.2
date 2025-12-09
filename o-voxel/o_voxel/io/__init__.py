from typing import Dict, Union
import torch
from .ply import *
from .npz import *
from .vxz import *


def read(file_path: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Read a file containing voxels.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        torch.Tensor: the coordinates of the voxels.
        Dict[str, torch.Tensor]: the attributes of the voxels.
    """
    if file_path.endswith('.npz'):
        return read_npz(file_path)
    elif file_path.endswith('.ply'):
        return read_ply(file_path)
    elif file_path.endswith('.vxz'):
        return read_vxz(file_path)
    else:
        raise ValueError(f"Unsupported file type {file_path}")
    
    
def write(file_path: str, coord: torch.Tensor, attr: Dict[str, torch.Tensor], **kwargs):
    """
    Write a file containing voxels.
    
    Args:
        file_path: Path to the file.
        coord: the coordinates of the voxels.
        attr: the attributes of the voxels.
    """
    if file_path.endswith('.npz'):
        write_npz(file_path, coord, attr, **kwargs)
    elif file_path.endswith('.ply'):
        write_ply(file_path, coord, attr, **kwargs)
    elif file_path.endswith('.vxz'):
        write_vxz(file_path, coord, attr, **kwargs)
    else:
        raise ValueError(f"Unsupported file type {file_path}")
