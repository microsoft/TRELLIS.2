from typing import *
import torch
import numpy as np


__all__ = [
    "read_npz",
    "write_npz",
]


def read_npz(file) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Read a NPZ file containing voxels.
    
    Args:
        file_path: Path or file object from which to read the NPZ file.
        
    Returns:
        torch.Tensor: the coordinates of the voxels.
        Dict[str, torch.Tensor]: the attributes of the voxels.
    """
    data = np.load(file)
    coord = torch.from_numpy(data['coord']).int()
    attr = {k: torch.from_numpy(v) for k, v in data.items() if k!= 'coord'}
    return coord, attr


def write_npz(file, coord: torch.Tensor, attr: Dict[str, torch.Tensor], compress=True):
    """
    Write a NPZ file containing voxels.
    
    Args:
        file_path: Path or file object to which to write the NPZ file.
        coord: the coordinates of the voxels.
        attr: the attributes of the voxels.
    """
    data = {'coord': coord.cpu().numpy().astype(np.uint16)}
    data.update({k: v.cpu().numpy() for k, v in attr.items()})
    if compress:
        np.savez_compressed(file, **data)
    else:
        np.savez(file, **data)
