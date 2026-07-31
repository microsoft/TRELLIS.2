import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from mlx_backend.attention import MlxMultiHeadAttention
from mlx_backend.norm import LayerNorm32
from mlx_backend.rope import apply_rope
from mlx_backend.sparse_conv import MlxSparseConv3d
from mlx_backend.sparse_tensor import MlxSparseTensor


def _numpy(value):
    mx.eval(value)
    return np.asarray(value)


def test_layer_norm_matches_torch_on_small_float32_input():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(3, 4, 8)).astype(np.float32)
    weight = rng.normal(size=(8,)).astype(np.float32)
    bias = rng.normal(size=(8,)).astype(np.float32)

    layer = LayerNorm32(8, eps=1e-6)
    layer.weight = mx.array(weight)
    layer.bias = mx.array(bias)
    actual = _numpy(layer(mx.array(values)))
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(values),
        (8,),
        torch.from_numpy(weight),
        torch.from_numpy(bias),
        eps=1e-6,
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_rope_matches_complex_pair_reference():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(5, 2, 8)).astype(np.float32)
    phases = rng.normal(size=(5, 4)).astype(np.float32)
    cos = np.cos(phases)
    sin = np.sin(phases)

    actual = _numpy(apply_rope(mx.array(values), mx.array(cos), mx.array(sin)))
    paired = values.reshape(5, 2, 4, 2)
    expected = np.empty_like(paired)
    expected[..., 0] = paired[..., 0] * cos[:, None] - paired[..., 1] * sin[:, None]
    expected[..., 1] = paired[..., 0] * sin[:, None] + paired[..., 1] * cos[:, None]
    np.testing.assert_allclose(actual, expected.reshape(values.shape), rtol=1e-6, atol=1e-6)


def test_mlx_sdpa_matches_torch():
    rng = np.random.default_rng(11)
    q = rng.normal(size=(6, 2, 4)).astype(np.float32)
    k = rng.normal(size=(6, 2, 4)).astype(np.float32)
    v = rng.normal(size=(6, 2, 4)).astype(np.float32)
    attention = MlxMultiHeadAttention(channels=8, num_heads=2)

    actual = _numpy(attention._sdpa(mx.array(q), mx.array(k), mx.array(v)))
    expected = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q).transpose(0, 1).unsqueeze(0),
        torch.from_numpy(k).transpose(0, 1).unsqueeze(0),
        torch.from_numpy(v).transpose(0, 1).unsqueeze(0),
        scale=0.5,
    )[0].transpose(0, 1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_sparse_conv_matches_numpy_reference():
    rng = np.random.default_rng(19)
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    feats = rng.normal(size=(len(coords), 2)).astype(np.float32)
    weight = rng.normal(size=(3, 3, 3, 3, 2)).astype(np.float32)
    bias = rng.normal(size=(3,)).astype(np.float32)

    layer = MlxSparseConv3d(2, 3, kernel_size=3)
    layer.weight = mx.array(weight)
    layer.bias = mx.array(bias)
    sparse = MlxSparseTensor(mx.array(feats), mx.array(coords), shape=(1, 2))
    actual = _numpy(layer(sparse).feats)

    lookup = {tuple(coord): index for index, coord in enumerate(coords)}
    expected = np.tile(bias, (len(coords), 1))
    offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    flat_weight = weight.reshape(3, 27, 2).transpose(1, 2, 0)
    for output_index, coord in enumerate(coords):
        for kernel_index, offset in enumerate(offsets):
            neighbor = (coord[0], coord[1] + offset[0], coord[2] + offset[1], coord[3] + offset[2])
            if neighbor in lookup:
                expected[output_index] += feats[lookup[neighbor]] @ flat_weight[kernel_index]

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
