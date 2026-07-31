import torch
import torch.nn.functional as F

from trellis2.backends import fp32_decode_thresholds_enabled
from trellis2.models.sc_vaes.sparse_unet_vae import _subdiv_logits
from trellis2.modules import sparse as sp


def test_fp32_decode_thresholds_env(monkeypatch):
    monkeypatch.setenv("TRELLIS_FP32_DECODE_THRESHOLDS", "0")
    assert fp32_decode_thresholds_enabled() is False

    monkeypatch.setenv("TRELLIS_FP32_DECODE_THRESHOLDS", "1")
    assert fp32_decode_thresholds_enabled() is True


def test_subdiv_logits_flag_off_matches_fp16_module(monkeypatch):
    monkeypatch.setenv("TRELLIS_FP32_DECODE_THRESHOLDS", "0")
    lin = sp.SparseLinear(4, 8).half()
    x = sp.SparseTensor(
        feats=torch.randn(5, 4).half(),
        coords=torch.zeros(5, 4, dtype=torch.int32),
    )

    result = _subdiv_logits(lin, x)
    expected = lin(x)

    assert result.feats.dtype == torch.float16
    assert torch.equal(result.feats, expected.feats)


def test_subdiv_logits_flag_on_computes_fp32(monkeypatch):
    monkeypatch.setenv("TRELLIS_FP32_DECODE_THRESHOLDS", "1")
    lin = sp.SparseLinear(4, 8).half()
    x = sp.SparseTensor(
        feats=torch.randn(5, 4).half(),
        coords=torch.zeros(5, 4, dtype=torch.int32),
    )

    result = _subdiv_logits(lin, x)
    expected = F.linear(
        x.feats.float(), lin.weight.float(), lin.bias.float()
    )

    assert result.feats.dtype == torch.float32
    assert torch.allclose(result.feats, expected)
    assert torch.equal(result.coords, x.coords)


def test_subdiv_logits_flag_on_preserves_fp32_output(monkeypatch):
    monkeypatch.setenv("TRELLIS_FP32_DECODE_THRESHOLDS", "1")
    lin = sp.SparseLinear(4, 8)
    x = sp.SparseTensor(
        feats=torch.randn(5, 4),
        coords=torch.zeros(5, 4, dtype=torch.int32),
    )

    result = _subdiv_logits(lin, x)
    expected = lin(x)

    assert torch.equal(result.feats, expected.feats)
