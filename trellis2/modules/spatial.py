import torch


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_, H*scale_factor, W*scale_factor, D*scale_factor)
    return x


def patchify(x: torch.Tensor, patch_size: int):
    """
    Patchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2
    for d in range(2, DIM + 2):
        assert x.shape[d] % patch_size == 0, f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"

    x = x.reshape(*x.shape[:2], *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], []))
    x = x.permute(0, 1, *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)]))
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size ** DIM), *(x.shape[-DIM:]))
    return x


def unpatchify(x: torch.Tensor, patch_size: int):
    """
    Unpatchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2
    assert x.shape[1] % (patch_size ** DIM) == 0, f"Second dimension of input tensor must be divisible by patch size to unpatchify, got {x.shape[1]} and {patch_size ** DIM}"

    x = x.reshape(x.shape[0], x.shape[1] // (patch_size ** DIM), *([patch_size] * DIM), *(x.shape[-DIM:]))
    x = x.permute(0, 1, *(sum([[2 + DIM + i, 2 + i] for i in range(DIM)], [])))
    x = x.reshape(x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(DIM)])
    return x
