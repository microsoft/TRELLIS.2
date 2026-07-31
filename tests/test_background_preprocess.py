import numpy as np
import pytest
from PIL import Image

from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline


class _FakeRembg:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        rgba = image.convert("RGBA")
        alpha = np.zeros((image.height, image.width), dtype=np.uint8)
        alpha[2:6, 2:6] = 255
        rgba.putalpha(Image.fromarray(alpha, mode="L"))
        return rgba


def _pipeline(rembg):
    pipeline = object.__new__(Trellis2ImageTo3DPipeline)
    pipeline.low_vram = False
    pipeline.rembg_model = rembg
    pipeline._device = "cpu"
    return pipeline


def test_rgba_with_transparency_bypasses_rmbg():
    rembg = _FakeRembg()
    rgba = np.full((8, 8, 4), 255, dtype=np.uint8)
    rgba[:, :, :3] = [80, 120, 160]
    rgba[:2, :, 3] = 0

    output = _pipeline(rembg).preprocess_image(Image.fromarray(rgba, mode="RGBA"))

    assert rembg.calls == 0
    assert output.mode == "RGB"
    assert output.size[0] > 0 and output.size[1] > 0


def test_opaque_rgba_uses_official_background_remover():
    rembg = _FakeRembg()
    opaque = Image.new("RGBA", (8, 8), (80, 120, 160, 255))

    output = _pipeline(rembg).preprocess_image(opaque)

    assert rembg.calls == 1
    assert output.mode == "RGB"
    assert output.size[0] > 0 and output.size[1] > 0


def test_soft_authored_alpha_is_preserved_without_rmbg():
    rembg = _FakeRembg()
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[2:6, 2:6, :3] = [80, 120, 160]
    rgba[2:6, 2:6, 3] = 128

    output = _pipeline(rembg).preprocess_image(Image.fromarray(rgba, mode="RGBA"))

    assert rembg.calls == 0
    assert output.size[0] > 0 and output.size[1] > 0


def test_empty_authored_alpha_has_actionable_error():
    rembg = _FakeRembg()
    empty = Image.new("RGBA", (8, 8), (80, 120, 160, 0))

    with pytest.raises(RuntimeError, match="empty alpha mask"):
        _pipeline(rembg).preprocess_image(empty)

    assert rembg.calls == 0
