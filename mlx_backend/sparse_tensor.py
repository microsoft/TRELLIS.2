"""
MLX sparse tensor representation.
Mirrors trellis2.modules.sparse.basic.SparseTensor but uses mx.array.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from fractions import Fraction
import mlx.core as mx


@dataclass
class MlxSparseTensor:
    """
    Sparse tensor with MLX arrays.

    Fields:
        feats: (N, C) feature matrix
        coords: (N, 4) int32 coordinates [batch, x, y, z]
        shape: (B, C) batch shape
        _scale: coordinate scale factors (for cache keying)
        _spatial_cache: nested dict keyed by scale, then by cache name
    """
    feats: mx.array
    coords: mx.array
    shape: Optional[Tuple[int, ...]] = None
    _scale: Tuple = (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1))
    _spatial_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.shape is None:
            batch_max = int(self.coords[:, 0].max().item()) + 1
            self.shape = (batch_max, *self.feats.shape[1:])

    @property
    def device(self):
        return self.feats.dtype  # MLX doesn't have device, everything is on GPU

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def N(self) -> int:
        return self.feats.shape[0]

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        cached = self.get_spatial_cache('shape')
        if cached is not None:
            return cached
        maxes = self.coords[:, 1:].max(axis=0)
        ss = tuple((int(m) + 1) for m in maxes.tolist())
        self.register_spatial_cache('shape', ss)
        return ss

    @property
    def layout(self) -> List[slice]:
        cached = self.get_spatial_cache('layout')
        if cached is not None:
            return cached
        batch_size = self.shape[0]
        batch_ids = self.coords[:, 0]
        layout = []
        start = 0
        for b in range(batch_size):
            count = int(mx.sum(batch_ids == b).item())
            layout.append(slice(start, start + count))
            start += count
        self.register_spatial_cache('layout', layout)
        return layout

    def replace(self, feats: mx.array, coords: Optional[mx.array] = None) -> 'MlxSparseTensor':
        """Create a new MlxSparseTensor with new features (and optionally new coords)."""
        new_shape = (self.shape[0], *feats.shape[1:]) if self.shape is not None else None
        return MlxSparseTensor(
            feats=feats,
            coords=coords if coords is not None else self.coords,
            shape=new_shape,
            _scale=self._scale,
            _spatial_cache=self._spatial_cache,
        )

    def register_spatial_cache(self, key: str, value: Any) -> None:
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key: str = None) -> Any:
        scale_key = str(self._scale)
        cur = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur
        return cur.get(key, None)

    def __repr__(self):
        return f"MlxSparseTensor(N={self.N}, shape={self.shape}, dtype={self.dtype})"


def mlx_sparse_cat(inputs: List[MlxSparseTensor], dim: int = -1) -> MlxSparseTensor:
    """Concatenate sparse tensors along feature dimension."""
    if dim == -1 or dim == 1:
        feats = mx.concatenate([s.feats for s in inputs], axis=-1)
        return inputs[0].replace(feats)
    elif dim == 0:
        # Batch concatenation: shift batch indices
        offset = 0
        all_coords = []
        all_feats = []
        for s in inputs:
            coords = mx.array(s.coords)  # copy
            coords = mx.concatenate([
                coords[:, :1] + offset,
                coords[:, 1:]
            ], axis=-1)
            all_coords.append(coords)
            all_feats.append(s.feats)
            offset += s.shape[0]
        return MlxSparseTensor(
            feats=mx.concatenate(all_feats, axis=0),
            coords=mx.concatenate(all_coords, axis=0),
        )
    else:
        raise ValueError(f"Unsupported dim={dim}")
