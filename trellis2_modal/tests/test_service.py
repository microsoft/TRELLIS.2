"""
Tests for TRELLIS2Service request parsing and validation.

These tests verify the request parsing and validation logic without
requiring GPU or the full TRELLIS.2 pipeline. They use mocks for
external dependencies.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


class TestParseGenerateRequest:
    """Tests for _parse_generate_request function."""

    @pytest.fixture
    def valid_image_b64(self):
        """Create a valid base64-encoded PNG image."""
        img = Image.new("RGB", (100, 100), color="red")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def test_missing_image_returns_error(self):
        """Missing 'image' field returns validation error."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request({"seed": 42})
        assert "error" in result
        assert result["error"]["code"] == "validation_error"
        assert "image" in result["error"]["message"].lower()

    def test_invalid_json_type_returns_error(self):
        """Non-dict request returns validation error."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request("not a dict")
        assert "error" in result
        assert result["error"]["code"] == "validation_error"

    def test_invalid_base64_returns_error(self):
        """Invalid base64 returns validation error."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request({"image": "not-valid-base64!!!"})
        assert "error" in result
        assert result["error"]["code"] == "validation_error"

    def test_corrupt_image_returns_error(self):
        """Valid base64 but corrupt image data returns error."""
        from trellis2_modal.service.service import _parse_generate_request

        garbage = base64.b64encode(b"not an image").decode("utf-8")
        result = _parse_generate_request({"image": garbage})
        assert "error" in result
        assert result["error"]["code"] == "validation_error"

    def test_valid_image_returns_params(self, valid_image_b64):
        """Valid image returns GenerateParams dataclass."""
        from trellis2_modal.service.service import (
            GenerateParams,
            _parse_generate_request,
        )

        result = _parse_generate_request({"image": valid_image_b64})
        assert isinstance(result, GenerateParams)
        assert result.image.size == (100, 100)

    def test_default_seed(self, valid_image_b64):
        """Default seed is 42."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request({"image": valid_image_b64})
        assert result.seed == 42

    def test_custom_seed(self, valid_image_b64):
        """Custom seed is preserved."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request({"image": valid_image_b64, "seed": 123})
        assert result.seed == 123

    def test_default_pipeline_type(self, valid_image_b64):
        """Default pipeline_type is 1024_cascade."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request({"image": valid_image_b64})
        assert result.pipeline_type == "1024_cascade"

    @pytest.mark.parametrize(
        "pipeline_type", ["512", "1024", "1024_cascade", "1536_cascade"]
    )
    def test_valid_pipeline_types(self, valid_image_b64, pipeline_type):
        """All 4 pipeline types are accepted."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request(
            {"image": valid_image_b64, "pipeline_type": pipeline_type}
        )
        assert result.pipeline_type == pipeline_type

    def test_invalid_pipeline_type_returns_error(self, valid_image_b64):
        """Invalid pipeline_type returns validation error."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request(
            {"image": valid_image_b64, "pipeline_type": "invalid"}
        )
        assert "error" in result
        assert result["error"]["code"] == "validation_error"
        assert "pipeline_type" in result["error"]["message"]

    def test_default_sampler_params(self, valid_image_b64):
        """Sampler params have correct defaults."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request({"image": valid_image_b64})
        assert result.ss_sampling_steps == 12
        assert result.ss_guidance_strength == 7.5
        assert result.shape_slat_sampling_steps == 12
        assert result.shape_slat_guidance_strength == 7.5
        assert result.tex_slat_sampling_steps == 12
        assert result.tex_slat_guidance_strength == 1.0

    def test_custom_sampler_params(self, valid_image_b64):
        """Custom sampler params are preserved."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request(
            {
                "image": valid_image_b64,
                "ss_sampling_steps": 8,
                "ss_guidance_strength": 5.0,
                "shape_slat_sampling_steps": 10,
                "shape_slat_guidance_strength": 6.0,
                "tex_slat_sampling_steps": 6,
                "tex_slat_guidance_strength": 2.0,
            }
        )
        assert result.ss_sampling_steps == 8
        assert result.ss_guidance_strength == 5.0
        assert result.shape_slat_sampling_steps == 10
        assert result.shape_slat_guidance_strength == 6.0
        assert result.tex_slat_sampling_steps == 6
        assert result.tex_slat_guidance_strength == 2.0

    def test_invalid_numeric_param_returns_error(self, valid_image_b64):
        """Invalid numeric param returns validation error."""
        from trellis2_modal.service.service import _parse_generate_request

        result = _parse_generate_request(
            {
                "image": valid_image_b64,
                "seed": "not a number",
            }
        )
        assert "error" in result
        assert result["error"]["code"] == "validation_error"


class TestParseExtractGLBRequest:
    """Tests for _parse_extract_glb_request function."""

    @pytest.fixture
    def valid_state_b64(self):
        """Create a valid base64-encoded compressed state."""
        from trellis2_modal.client.compression import compress_state
        import numpy as np

        state = {
            "vertices": np.array([[0, 0, 0]], dtype=np.float32),
            "faces": np.array([[0, 0, 0]], dtype=np.int32),
        }
        compressed = compress_state(state)
        return base64.b64encode(compressed).decode("utf-8")

    def test_missing_state_returns_error(self):
        """Missing 'state' field returns validation error."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request({})
        assert "error" in result
        assert result["error"]["code"] == "validation_error"
        assert "state" in result["error"]["message"].lower()

    def test_invalid_json_type_returns_error(self):
        """Non-dict request returns validation error."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request("not a dict")
        assert "error" in result
        assert result["error"]["code"] == "validation_error"

    def test_invalid_state_returns_error(self):
        """Invalid state data returns validation error."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        garbage = base64.b64encode(b"not valid state").decode("utf-8")
        result = _parse_extract_glb_request({"state": garbage})
        assert "error" in result
        assert result["error"]["code"] == "validation_error"

    def test_valid_state_returns_params(self, valid_state_b64):
        """Valid state returns ExtractGLBParams dataclass."""
        from trellis2_modal.service.service import (
            ExtractGLBParams,
            _parse_extract_glb_request,
        )

        result = _parse_extract_glb_request({"state": valid_state_b64})
        assert isinstance(result, ExtractGLBParams)
        assert "vertices" in result.state

    def test_default_decimation_target(self, valid_state_b64):
        """Default decimation_target is 1000000."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request({"state": valid_state_b64})
        assert result.decimation_target == 1000000

    def test_default_texture_size(self, valid_state_b64):
        """Default texture_size is 4096."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request({"state": valid_state_b64})
        assert result.texture_size == 4096

    def test_default_remesh(self, valid_state_b64):
        """Default remesh is True."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request({"state": valid_state_b64})
        assert result.remesh is True

    def test_custom_decimation_target(self, valid_state_b64):
        """Custom decimation_target is preserved."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request(
            {
                "state": valid_state_b64,
                "decimation_target": 500000,
            }
        )
        assert result.decimation_target == 500000

    def test_decimation_target_too_small_returns_error(self, valid_state_b64):
        """decimation_target < 1000 returns error."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request(
            {
                "state": valid_state_b64,
                "decimation_target": 500,
            }
        )
        assert "error" in result
        assert result["error"]["code"] == "validation_error"

    @pytest.mark.parametrize("texture_size", [512, 1024, 2048, 4096])
    def test_valid_texture_sizes(self, valid_state_b64, texture_size):
        """Valid texture sizes are accepted."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request(
            {
                "state": valid_state_b64,
                "texture_size": texture_size,
            }
        )
        assert result.texture_size == texture_size

    def test_invalid_texture_size_returns_error(self, valid_state_b64):
        """Invalid texture_size returns error."""
        from trellis2_modal.service.service import _parse_extract_glb_request

        result = _parse_extract_glb_request(
            {
                "state": valid_state_b64,
                "texture_size": 999,
            }
        )
        assert "error" in result
        assert result["error"]["code"] == "validation_error"

# Note: Authentication is now handled by Modal Proxy Auth at the proxy level.
# No _check_auth_or_error tests needed since requests that reach endpoints are pre-authenticated.


class TestGenerateRequestId:
    """Tests for generate_request_id function."""

    def test_returns_string(self):
        """Returns a string."""
        from trellis2_modal.service.service import generate_request_id

        result = generate_request_id()
        assert isinstance(result, str)

    def test_starts_with_req(self):
        """Starts with 'req_' prefix."""
        from trellis2_modal.service.service import generate_request_id

        result = generate_request_id()
        assert result.startswith("req_")

    def test_unique_ids(self):
        """Generated IDs are unique."""
        from trellis2_modal.service.service import generate_request_id

        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100


class TestErrorResponse:
    """Tests for _error_response function."""

    def test_returns_correct_structure(self):
        """Returns dict with error.code and error.message."""
        from trellis2_modal.service.service import _error_response

        result = _error_response("test_code", "test message")
        assert result == {
            "error": {
                "code": "test_code",
                "message": "test message",
            }
        }
