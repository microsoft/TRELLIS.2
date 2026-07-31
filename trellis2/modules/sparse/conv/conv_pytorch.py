"""
Pure PyTorch submanifold sparse conv3d backend.

Algorithm for 3×3×3 submanifold sparse conv:
1. Hash all voxel coords → flat index (b×S³ + x×S² + y×S + z)
2. For each of 27 kernel offsets: shift coords, lookup neighbors via hash
3. Gather neighbor features, matmul with kernel weight per offset, scatter_add
4. Cache neighbor maps (topology constant during inference)
"""
import math
import itertools
import torch
import torch.nn as nn
from .. import SparseTensor


def _build_neighbor_map(coords: torch.Tensor, shape: torch.Size, spatial_shape: torch.Size,
                        kernel_size: tuple, dilation: tuple, device: torch.device):
    """
    Build neighbor map for submanifold sparse conv.

    Returns:
        neighbor_map: (K, N) int64 tensor. For each kernel offset k and voxel i,
                      neighbor_map[k, i] = index of neighbor in coords, or -1 if absent.
    """
    N = coords.shape[0]
    batch_size = shape[0]
    D, H, W = spatial_shape

    # Ensure coords are on the target device
    if coords.device != device:
        coords = coords.to(device)

    # Build lookup table: flat_coord -> index in coords tensor
    # Flat key = batch * (D*H*W) + x*H*W + y*W + z
    DHW = D * H * W
    flat_keys = (coords[:, 0].long() * DHW +
                 coords[:, 1].long() * (H * W) +
                 coords[:, 2].long() * W +
                 coords[:, 3].long())
    table_size = batch_size * DHW
    lookup = torch.full((table_size,), -1, dtype=torch.long, device=device)
    lookup[flat_keys] = torch.arange(N, dtype=torch.long, device=device)

    # Generate kernel offsets
    kd, kh, kw = kernel_size
    dd, dh, dw = dilation
    offsets = []
    for dx, dy, dz in itertools.product(
        range(-(kd // 2), kd // 2 + 1),
        range(-(kh // 2), kh // 2 + 1),
        range(-(kw // 2), kw // 2 + 1),
    ):
        offsets.append((dx * dd, dy * dh, dz * dw))
    K = len(offsets)
    offsets_tensor = torch.tensor(offsets, dtype=coords.dtype, device=device)  # (K, 3)

    # For each offset, shift coords and look up
    neighbor_map = torch.full((K, N), -1, dtype=torch.long, device=device)
    for k_idx in range(K):
        shifted = coords.clone()
        shifted[:, 1] = shifted[:, 1] + offsets_tensor[k_idx, 0]
        shifted[:, 2] = shifted[:, 2] + offsets_tensor[k_idx, 1]
        shifted[:, 3] = shifted[:, 3] + offsets_tensor[k_idx, 2]

        # Bounds check
        valid = ((shifted[:, 1] >= 0) & (shifted[:, 1] < D) &
                 (shifted[:, 2] >= 0) & (shifted[:, 2] < H) &
                 (shifted[:, 3] >= 0) & (shifted[:, 3] < W))

        flat_shifted = (shifted[:, 0].long() * DHW +
                        shifted[:, 1].long() * (H * W) +
                        shifted[:, 2].long() * W +
                        shifted[:, 3].long())
        # Clamp for safe indexing (invalid entries will be masked out)
        flat_shifted = flat_shifted.clamp(0, table_size - 1)
        looked_up = lookup[flat_shifted]
        looked_up[~valid] = -1
        neighbor_map[k_idx] = looked_up

    return neighbor_map


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1,
                       dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), \
        'PyTorch backend only supports submanifold sparse convolution (stride=1, padding=None)'

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

    # Store weight in same format as flex_gemm: (Co, Kd, Kh, Kw, Ci)
    # This ensures checkpoint compatibility — flex_gemm permutes (Co, Ci, K...) -> (Co, K..., Ci) at init
    self.weight = nn.Parameter(torch.empty((out_channels, *self.kernel_size, in_channels)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # Initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
            self.weight.permute(0, 4, 1, 2, 3))  # back to (Co, Ci, K...) for fan calc
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)


def _get_weight_kernel(self):
    """Reshape stored weight (Co, Kd, Kh, Kw, Ci) -> (K, Ci, Co) for gather-matmul."""
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    K = Kd * Kh * Kw
    # (Co, Kd, Kh, Kw, Ci) -> (Co, K, Ci) -> (K, Ci, Co)
    return self.weight.reshape(Co, K, Ci).permute(1, 2, 0)


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    N = x.feats.shape[0]

    # Check cache for neighbor map
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    cache_key = f'SubMConv3d_neighbor_cache_pytorch_{self.kernel_size}_dilation{self.dilation}'
    neighbor_map = x.get_spatial_cache(cache_key)

    if neighbor_map is None:
        neighbor_map = _build_neighbor_map(
            x.coords, x.shape, x.spatial_shape,
            self.kernel_size, self.dilation, x.device,
        )
        x.register_spatial_cache(cache_key, neighbor_map)

    K = neighbor_map.shape[0]
    w = _get_weight_kernel(self)  # (K, Ci, Co)

    # Pad feats with a zero row for -1 indices
    feats_padded = torch.cat([x.feats, torch.zeros(1, x.feats.shape[1], device=x.device, dtype=x.feats.dtype)], dim=0)
    pad_idx = N

    # Replace -1 with pad_idx
    safe_map = neighbor_map.clone()
    safe_map[safe_map < 0] = pad_idx  # (K, N)

    # Gather: (K, N, Ci)
    gathered = feats_padded[safe_map]

    # Matmul per kernel offset: (K, N, Ci) @ (K, Ci, Co) -> (K, N, Co)
    out = torch.bmm(gathered, w.to(gathered.dtype))

    # Mask out invalid neighbors before summing
    valid_mask = (neighbor_map >= 0).unsqueeze(-1)  # (K, N, 1)
    out = out * valid_mask

    # Sum over kernel offsets
    result = out.sum(dim=0)  # (N, Co)

    if self.bias is not None:
        result = result + self.bias

    return x.replace(result)


def sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1,
                                dilation=1, bias=True, indice_key=None):
    sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1,
                       dilation=dilation, padding=None, bias=bias, indice_key=indice_key)


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    # For submanifold case (stride=1), inverse conv is the same as forward conv
    return sparse_conv3d_forward(self, x)
