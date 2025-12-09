from typing import *
from fractions import Fraction
import torch
from . import config


__all__ = [
    'VarLenTensor',
    'varlen_cat',
    'varlen_unbind',
    'SparseTensor',
    'sparse_cat',
    'sparse_unbind',
]


class VarLenTensor:
    """
    Sequential tensor with variable length.
    
    Args:
        feats (torch.Tensor): Features of the varlen tensor.
        layout (List[slice]): Layout of the varlen tensor for each batch
    """
    def __init__(self, feats: torch.Tensor, layout: List[slice]=None):
        self.feats = feats
        self.layout = layout if layout is not None else [slice(0, feats.shape[0])]
        self._cache = {}
        
    @staticmethod
    def layout_from_seqlen(seqlen: list) -> List[slice]:
        """
        Create a layout from a tensor of sequence lengths.
        """
        layout = []
        start = 0
        for l in seqlen:
            layout.append(slice(start, start + l))
            start += l
        return layout
        
    @staticmethod
    def from_tensor_list(tensor_list: List[torch.Tensor]) -> 'VarLenTensor':
        """
        Create a VarLenTensor from a list of tensors.
        """
        feats = torch.cat(tensor_list, dim=0)
        layout = []
        start = 0
        for tensor in tensor_list:
            layout.append(slice(start, start + tensor.shape[0]))
            start += tensor.shape[0]
        return VarLenTensor(feats, layout)
    
    def to_tensor_list(self) -> List[torch.Tensor]:
        """
        Convert a VarLenTensor to a list of tensors.
        """
        tensor_list = []
        for s in self.layout:
            tensor_list.append(self.feats[s])
        return tensor_list
    
    def __len__(self) -> int:
        return len(self.layout)
    
    @property
    def shape(self) -> torch.Size:
        return torch.Size([len(self.layout), *self.feats.shape[1:]])
    
    def dim(self) -> int:
        return len(self.shape)
    
    @property
    def ndim(self) -> int:
        return self.dim()

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device
    
    @property
    def seqlen(self) -> torch.LongTensor:
        if 'seqlen' not in self._cache:
            self._cache['seqlen'] = torch.tensor([l.stop - l.start for l in self.layout], dtype=torch.long, device=self.device)
        return self._cache['seqlen']
    
    @property
    def cum_seqlen(self) -> torch.LongTensor:
        if 'cum_seqlen' not in self._cache:
            self._cache['cum_seqlen'] = torch.cat([
                torch.tensor([0], dtype=torch.long, device=self.device),
                self.seqlen.cumsum(dim=0)
            ], dim=0)
        return self._cache['cum_seqlen']
    
    @property
    def batch_boardcast_map(self) -> torch.LongTensor:
        """
        Get the broadcast map for the varlen tensor.
        """
        if 'batch_boardcast_map' not in self._cache:
            self._cache['batch_boardcast_map'] = torch.repeat_interleave(
                torch.arange(len(self.layout), device=self.device),
                self.seqlen,
            )
        return self._cache['batch_boardcast_map']
    
    @overload
    def to(self, dtype: torch.dtype, *, non_blocking: bool = False, copy: bool = False) -> 'VarLenTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, *, non_blocking: bool = False, copy: bool = False) -> 'VarLenTensor': ...

    def to(self, *args, **kwargs) -> 'VarLenTensor':
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)
        
        new_feats = self.feats.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        return self.replace(new_feats)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'VarLenTensor':
        new_feats = self.feats.cpu()
        return self.replace(new_feats)
    
    def cuda(self) -> 'VarLenTensor':
        new_feats = self.feats.cuda()
        return self.replace(new_feats)

    def half(self) -> 'VarLenTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)
    
    def float(self) -> 'VarLenTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)
    
    def detach(self) -> 'VarLenTensor':
        new_feats = self.feats.detach()
        return self.replace(new_feats)

    def reshape(self, *shape) -> 'VarLenTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)
    
    def unbind(self, dim: int) -> List['VarLenTensor']:
        return varlen_unbind(self, dim)

    def replace(self, feats: torch.Tensor) -> 'VarLenTensor':
        new_tensor = VarLenTensor(
            feats=feats,
            layout=self.layout,
        )
        new_tensor._cache = self._cache
        return new_tensor
    
    def to_dense(self, max_length=None) -> torch.Tensor:
        """
        Convert a VarLenTensor to a dense representation without for-loop.
        
        Returns:
            dense (torch.Tensor): (N, L, C) dense tensor
            mask (torch.BoolTensor): (N, L) mask indicating valid positions
        """
        N = len(self)
        L = max_length or self.seqlen.max().item()
        spatial = self.feats.shape[1:]
        idx = torch.arange(L, device=self.device).unsqueeze(0).expand(N, L)
        mask = (idx < self.seqlen.unsqueeze(1))
        mapping = mask.reshape(-1).cumsum(dim=0) - 1
        dense = self.feats[mapping]
        dense = dense.reshape(N, L, *spatial)
        return dense, mask

    def __neg__(self) -> 'VarLenTensor':
        return self.replace(-self.feats)
    
    def __elemwise__(self, other: Union[torch.Tensor, 'VarLenTensor'], op: callable) -> 'VarLenTensor':
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = other[self.batch_boardcast_map]
            except:
                pass
        if isinstance(other, VarLenTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        return new_tensor

    def __add__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.add)
    
    def __sub__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.sub)
    
    def __rsub__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other: Union[torch.Tensor, 'VarLenTensor', float]) -> 'VarLenTensor':
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, list):
            assert all(isinstance(i, int) for i in idx), f"Only integer indices are supported: {idx}"
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")
        
        new_feats = []
        new_layout = []
        start = 0
        for new_idx, old_idx in enumerate(idx):
            new_feats.append(self.feats[self.layout[old_idx]])
            new_layout.append(slice(start, start + len(new_feats[-1])))
            start += len(new_feats[-1])
        new_feats = torch.cat(new_feats, dim=0).contiguous()
        new_tensor = VarLenTensor(feats=new_feats, layout=new_layout)
        return new_tensor
    
    def reduce(self, op: str, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        if isinstance(dim, int):
            dim = (dim,)
        
        if op =='mean':
            red = self.feats.mean(dim=dim, keepdim=keepdim)
        elif op =='sum':
            red = self.feats.sum(dim=dim, keepdim=keepdim)
        elif op == 'prod':
            red = self.feats.prod(dim=dim, keepdim=keepdim)
        else:
            raise ValueError(f"Unsupported reduce operation: {op}")
        
        if dim is None or 0 in dim:
            return red
        
        red = torch.segment_reduce(red, reduce=op, lengths=self.seqlen)
        return red
    
    def mean(self, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        return self.reduce(op='mean', dim=dim, keepdim=keepdim)
        
    def sum(self, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        return self.reduce(op='sum', dim=dim, keepdim=keepdim)
        
    def prod(self, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        return self.reduce(op='prod', dim=dim, keepdim=keepdim)
    
    def std(self, dim: Optional[Union[int, Tuple[int,...]]] = None, keepdim: bool = False) -> torch.Tensor:
        mean = self.mean(dim=dim, keepdim=True)
        mean2 = self.replace(self.feats ** 2).mean(dim=dim, keepdim=True)
        std = (mean2 - mean ** 2).sqrt()
        return std
    
    def __repr__(self) -> str:
        return f"VarLenTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"


def varlen_cat(inputs: List[VarLenTensor], dim: int = 0) -> VarLenTensor:
    """
    Concatenate a list of varlen tensors.
    
    Args:
        inputs (List[VarLenTensor]): List of varlen tensors to concatenate.
    """
    if dim == 0:
        new_feats = torch.cat([input.feats for input in inputs], dim=0)
        start = 0
        new_layout = []
        for input in inputs:
            for l in input.layout:
                new_layout.append(slice(start, start + l.stop - l.start))
                start += l.stop - l.start
        output = VarLenTensor(feats=new_feats, layout=new_layout)
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def varlen_unbind(input: VarLenTensor, dim: int) -> Union[List[VarLenTensor]]:
    """
    Unbind a varlen tensor along a dimension.
    
    Args:
        input (VarLenTensor): Varlen tensor to unbind.
        dim (int): Dimension to unbind.
    """
    if dim == 0:
        return [input[i] for i in range(len(input))]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]
    

class SparseTensor(VarLenTensor):
    """
    Sparse tensor with support for both torchsparse and spconv backends.
    
    Parameters:
    - feats (torch.Tensor): Features of the sparse tensor.
    - coords (torch.Tensor): Coordinates of the sparse tensor.
    - shape (torch.Size): Shape of the sparse tensor.
    - layout (List[slice]): Layout of the sparse tensor for each batch
    - data (SparseTensorData): Sparse tensor data used for convolusion

    NOTE:
    - Data corresponding to a same batch should be contiguous.
    - Coords should be in [0, 1023]
    """
    SparseTensorData = None

    @overload
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor, shape: Optional[torch.Size] = None, **kwargs): ...

    @overload
    def __init__(self, data, shape: Optional[torch.Size] = None, **kwargs): ...

    def __init__(self, *args, **kwargs):
        # Lazy import of sparse tensor backend
        if self.SparseTensorData is None:
            import importlib
            if config.CONV == 'torchsparse':
                self.SparseTensorData = importlib.import_module('torchsparse').SparseTensor
            elif config.CONV == 'spconv':
                self.SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor
                
        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if 'data' in kwargs else 0

        if method_id == 0:
            feats, coords, shape = args + (None,) * (3 - len(args))
            if 'feats' in kwargs:
                feats = kwargs['feats']
                del kwargs['feats']
            if 'coords' in kwargs:
                coords = kwargs['coords']
                del kwargs['coords']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']

            if config.CONV == 'torchsparse':
                self.data = self.SparseTensorData(feats, coords, **kwargs)
            elif config.CONV == 'spconv':
                spatial_shape = list(coords.max(0)[0] + 1)
                self.data = self.SparseTensorData(feats.reshape(feats.shape[0], -1), coords, spatial_shape[1:], spatial_shape[0], **kwargs)
                self.data._features = feats
            else:
                self.data = {
                    'feats': feats,
                    'coords': coords,
                }
        elif method_id == 1:
            data, shape = args + (None,) * (2 - len(args))
            if 'data' in kwargs:
                data = kwargs['data']
                del kwargs['data']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']

            self.data = data

        self._shape = shape
        self._scale = kwargs.get('scale', (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)))
        self._spatial_cache = kwargs.get('spatial_cache', {})

        if config.DEBUG:
            try:
                assert self.feats.shape[0] == self.coords.shape[0], f"Invalid feats shape: {self.feats.shape}, coords shape: {self.coords.shape}"
                assert self.shape == self.__cal_shape(self.feats, self.coords), f"Invalid shape: {self.shape}"
                assert self.layout == self.__cal_layout(self.coords, self.shape[0]), f"Invalid layout: {self.layout}"
                for i in range(self.shape[0]):
                    assert torch.all(self.coords[self.layout[i], 0] == i), f"The data of batch {i} is not contiguous"
            except Exception as e:
                print('Debugging information:')
                print(f"- Shape: {self.shape}")
                print(f"- Layout: {self.layout}")
                print(f"- Scale: {self._scale}")
                print(f"- Coords: {self.coords}")
                raise e
        
    @staticmethod
    def from_tensor_list(feats_list: List[torch.Tensor], coords_list: List[torch.Tensor]) -> 'SparseTensor':
        """
        Create a SparseTensor from a list of tensors.
        """
        feats = torch.cat(feats_list, dim=0)
        coords = []
        for i, coord in enumerate(coords_list):
            coord = torch.cat([torch.full_like(coord[:, :1], i), coord[:, 1:]], dim=1)
            coords.append(coord)
        coords = torch.cat(coords, dim=0)
        return SparseTensor(feats, coords)
    
    def to_tensor_list(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Convert a SparseTensor to list of tensors.
        """
        feats_list = []
        coords_list = []
        for s in self.layout:
            feats_list.append(self.feats[s])
            coords_list.append(self.coords[s])
        return feats_list, coords_list
    
    def __len__(self) -> int:
        return len(self.layout)
        
    def __cal_shape(self, feats, coords):
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)
    
    def __cal_layout(self, coords, batch_size):
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0) 
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]
        return layout
    
    def __cal_spatial_shape(self, coords):
        return torch.Size((coords[:, 1:].max(0)[0] + 1).tolist())
    
    @property
    def shape(self) -> torch.Size:
        if self._shape is None:
            self._shape = self.__cal_shape(self.feats, self.coords)
        return self._shape
    
    @property
    def layout(self) -> List[slice]:
        layout = self.get_spatial_cache('layout')
        if layout is None:
            layout = self.__cal_layout(self.coords, self.shape[0])
            self.register_spatial_cache('layout', layout)
        return layout
    
    @property
    def spatial_shape(self) -> torch.Size:
        spatial_shape = self.get_spatial_cache('shape')
        if spatial_shape is None:
            spatial_shape = self.__cal_spatial_shape(self.coords)
            self.register_spatial_cache('shape', spatial_shape)
        return spatial_shape

    @property
    def feats(self) -> torch.Tensor:
        if config.CONV == 'torchsparse':
            return self.data.F
        elif config.CONV == 'spconv':
            return self.data.features
        else:
            return self.data['feats']
    
    @feats.setter
    def feats(self, value: torch.Tensor):
        if config.CONV == 'torchsparse':
            self.data.F = value
        elif config.CONV == 'spconv':
            self.data.features = value
        else:
            self.data['feats'] = value

    @property
    def coords(self) -> torch.Tensor:
        if config.CONV == 'torchsparse':
            return self.data.C
        elif config.CONV == 'spconv':
            return self.data.indices
        else:
            return self.data['coords']
        
    @coords.setter
    def coords(self, value: torch.Tensor):
        if config.CONV == 'torchsparse':
            self.data.C = value
        elif config.CONV == 'spconv':
            self.data.indices = value
        else:
            self.data['coords'] = value

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device
    
    @property
    def seqlen(self) -> torch.LongTensor:
        seqlen = self.get_spatial_cache('seqlen')
        if seqlen is None:
            seqlen = torch.tensor([l.stop - l.start for l in self.layout], dtype=torch.long, device=self.device)
            self.register_spatial_cache('seqlen', seqlen)
        return seqlen
    
    @property
    def cum_seqlen(self) -> torch.LongTensor:
        cum_seqlen = self.get_spatial_cache('cum_seqlen')
        if cum_seqlen is None:
            cum_seqlen = torch.cat([
                torch.tensor([0], dtype=torch.long, device=self.device),
                self.seqlen.cumsum(dim=0)
            ], dim=0)
            self.register_spatial_cache('cum_seqlen', cum_seqlen)
        return cum_seqlen
    
    @property
    def batch_boardcast_map(self) -> torch.LongTensor:
        """
        Get the broadcast map for the varlen tensor.
        """
        batch_boardcast_map = self.get_spatial_cache('batch_boardcast_map')
        if batch_boardcast_map is None:
            batch_boardcast_map = torch.repeat_interleave(
                torch.arange(len(self.layout), device=self.device),
                self.seqlen,
            )
            self.register_spatial_cache('batch_boardcast_map', batch_boardcast_map)
        return batch_boardcast_map

    @overload
    def to(self, dtype: torch.dtype, *, non_blocking: bool = False, copy: bool = False) -> 'SparseTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, *, non_blocking: bool = False, copy: bool = False) -> 'SparseTensor': ...

    def to(self, *args, **kwargs) -> 'SparseTensor':
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)
        
        new_feats = self.feats.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        new_coords = self.coords.to(device=device, non_blocking=non_blocking, copy=copy)
        return self.replace(new_feats, new_coords)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'SparseTensor':
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        return self.replace(new_feats, new_coords)
    
    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)

    def half(self) -> 'SparseTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)
    
    def float(self) -> 'SparseTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)
    
    def detach(self) -> 'SparseTensor':
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        return self.replace(new_feats, new_coords)

    def reshape(self, *shape) -> 'SparseTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)
    
    def unbind(self, dim: int) -> List['SparseTensor']:
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> 'SparseTensor':
        if config.CONV == 'torchsparse':
            new_data = self.SparseTensorData(
                feats=feats,
                coords=self.data.coords if coords is None else coords,
                stride=self.data.stride,
                spatial_range=self.data.spatial_range,
            )
            new_data._caches = self.data._caches
        elif config.CONV == 'spconv':
            new_data = self.SparseTensorData(
                self.data.features.reshape(self.data.features.shape[0], -1),
                self.data.indices,
                self.data.spatial_shape,
                self.data.batch_size,
                self.data.grid,
                self.data.voxel_num,
                self.data.indice_dict
            )
            new_data._features = feats
            new_data.benchmark = self.data.benchmark
            new_data.benchmark_record = self.data.benchmark_record
            new_data.thrust_allocator = self.data.thrust_allocator
            new_data._timer = self.data._timer
            new_data.force_algo = self.data.force_algo
            new_data.int8_scale = self.data.int8_scale
            if coords is not None:
                new_data.indices = coords
        else:
            new_data = {
                'feats': feats,
                'coords': self.data['coords'] if coords is None else coords,
            }
        new_tensor = SparseTensor(
            new_data,
            shape=torch.Size([self._shape[0]] + list(feats.shape[1:])) if self._shape is not None else None,
            scale=self._scale,
            spatial_cache=self._spatial_cache
        )
        return new_tensor
    
    def to_dense(self) -> torch.Tensor:
        if config.CONV == 'torchsparse':
            return self.data.dense()
        elif config.CONV == 'spconv':
            return self.data.dense()
        else:
            spatial_shape = self.spatial_shape
            ret = torch.zeros(*self.shape, *spatial_shape, dtype=self.dtype, device=self.device)
            idx = [self.coords[:, 0], slice(None)] + self.coords[:, 1:].unbind(1)
            ret[tuple(idx)] = self.feats
            return ret

    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> 'SparseTensor':
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        coords = torch.cat([
            torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
            coords.repeat(N, 1),
        ], dim=1).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats=feats, coords=coords)

    def __merge_sparse_cache(self, other: 'SparseTensor') -> dict:
        new_cache = {}
        for k in set(list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k]
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k]
                else:
                    new_cache[k].update(other._spatial_cache[k])
        return new_cache
    
    def __elemwise__(self, other: Union[torch.Tensor, VarLenTensor], op: callable) -> 'SparseTensor':
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = other[self.batch_boardcast_map]
            except:
                pass
        if isinstance(other, VarLenTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, list):
            assert all(isinstance(i, int) for i in idx), f"Only integer indices are supported: {idx}"
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")
        
        new_coords = []
        new_feats = []
        new_layout = []
        new_shape = torch.Size([len(idx)] + list(self.shape[1:]))
        start = 0
        for new_idx, old_idx in enumerate(idx):
            new_coords.append(self.coords[self.layout[old_idx]].clone())
            new_coords[-1][:, 0] = new_idx
            new_feats.append(self.feats[self.layout[old_idx]])
            new_layout.append(slice(start, start + len(new_coords[-1])))
            start += len(new_coords[-1])
        new_coords = torch.cat(new_coords, dim=0).contiguous()
        new_feats = torch.cat(new_feats, dim=0).contiguous()
        new_tensor = SparseTensor(feats=new_feats, coords=new_coords, shape=new_shape)
        new_tensor.register_spatial_cache('layout', new_layout)
        return new_tensor
    
    def clear_spatial_cache(self) -> None:
        """
        Clear all spatial caches.
        """
        self._spatial_cache = {}

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)
    
    def __repr__(self) -> str:
        return f"SparseTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    """
    Concatenate a list of sparse tensors.
    
    Args:
        inputs (List[SparseTensor]): List of sparse tensors to concatenate.
    """
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    """
    Unbind a sparse tensor along a dimension.
    
    Args:
        input (SparseTensor): Sparse tensor to unbind.
        dim (int): Dimension to unbind.
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]
