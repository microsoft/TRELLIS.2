from typing import *
import torch
import torch.nn as nn
from .. import SparseTensor


class SparseSpatial2Channel(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as rearranging its features from spatial to channel.
    """
    def __init__(self, factor: int = 2):
        super(SparseSpatial2Channel, self).__init__()
        self.factor = factor

    def forward(self, x: SparseTensor) -> SparseTensor:
        DIM = x.coords.shape[-1] - 1
        cache = x.get_spatial_cache(f'spatial2channel_{self.factor}')
        if cache is None:
            coord = list(x.coords.unbind(dim=-1))
            for i in range(DIM):
                coord[i+1] = coord[i+1] // self.factor
            subidx = x.coords[:, 1:] % self.factor
            subidx = sum([subidx[..., i] * self.factor ** i for i in range(DIM)])

            MAX = [(s + self.factor - 1) // self.factor for s in x.spatial_shape]
            OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
            code = sum([c * o for c, o in zip(coord, OFFSET)])
            code, idx = code.unique(return_inverse=True)

            new_coords = torch.stack(
                [code // OFFSET[0]] +
                [(code // OFFSET[i+1]) % MAX[i] for i in range(DIM)],
                dim=-1
            )
        else:
            new_coords, idx, subidx = cache
            
        new_feats = torch.zeros(new_coords.shape[0] * self.factor ** DIM, x.feats.shape[1], device=x.feats.device, dtype=x.feats.dtype)
        new_feats[idx * self.factor ** DIM + subidx] = x.feats

        out = SparseTensor(new_feats.reshape(new_coords.shape[0], -1), new_coords, None if x._shape is None else torch.Size([x._shape[0], x._shape[1] * self.factor ** DIM]))
        out._scale = tuple([s * self.factor for s in x._scale])
        out._spatial_cache = x._spatial_cache
        
        if cache is None:
            x.register_spatial_cache(f'spatial2channel_{self.factor}', (new_coords, idx, subidx))
            out.register_spatial_cache(f'channel2spatial_{self.factor}', (x.coords, idx, subidx))
            out.register_spatial_cache(f'shape', torch.Size(MAX))
            if self.training:
                subdivision = torch.zeros((new_coords.shape[0], self.factor ** DIM), device=x.device, dtype=torch.bool)
                subdivision[idx, subidx] = True
                out.register_spatial_cache(f'subdivision', subdivision)
                
        return out


class SparseChannel2Spatial(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as rearranging its features from channel to spatial.
    """
    def __init__(self, factor: int = 2):
        super(SparseChannel2Spatial, self).__init__()
        self.factor = factor

    def forward(self, x: SparseTensor, subdivision: Optional[SparseTensor] = None) -> SparseTensor:
        DIM = x.coords.shape[-1] - 1

        cache = x.get_spatial_cache(f'channel2spatial_{self.factor}')
        if cache is None:
            if subdivision is None:
                raise ValueError('Cache not found. Provide subdivision tensor or pair SparseChannel2Spatial with SparseSpatial2Channel.')
            else:
                sub = subdivision.feats         # [N, self.factor ** DIM]
                N_leaf = sub.sum(dim=-1)        # [N]
                subidx = sub.nonzero()[:, -1]
                new_coords = x.coords.clone().detach()
                new_coords[:, 1:] *= self.factor
                new_coords = torch.repeat_interleave(new_coords, N_leaf, dim=0, output_size=subidx.shape[0])
                for i in range(DIM):
                    new_coords[:, i+1] += subidx // self.factor ** i % self.factor
                idx = torch.repeat_interleave(torch.arange(x.coords.shape[0], device=x.device), N_leaf, dim=0, output_size=subidx.shape[0])
        else:
            new_coords, idx, subidx = cache

        x_feats = x.feats.reshape(x.feats.shape[0] * self.factor ** DIM, -1)
        new_feats = x_feats[idx * self.factor ** DIM + subidx]
        out = SparseTensor(new_feats, new_coords, None if x._shape is None else torch.Size([x._shape[0], x._shape[1] // self.factor ** DIM]))
        out._scale = tuple([s / self.factor for s in x._scale])
        if cache is not None:           # only keep cache when subdiv following it
            out._spatial_cache = x._spatial_cache
        return out
