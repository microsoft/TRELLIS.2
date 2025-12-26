"""
Tests for the TRELLIS.2 API client.

Tests validate TRELLIS2APIClient HTTP calls, error handling, and response parsing.
"""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from trellis2_modal.client.api import APIError, TRELLIS2APIClient


class TestTRELLIS2APIClientInit:
    """Tests for TRELLIS2APIClient initialization."""

    def test_init_stores_credentials(self) -> None:
        """Client should store modal_key and modal_secret."""
        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test123",
            modal_secret="ws-secret456",
        )
        assert client.modal_key == "wk-test123"
        assert client.modal_secret == "ws-secret456"

    def test_init_derives_urls_for_path_routing(self) -> None:
        """Path-based URLs should derive generate and extract_glb URLs."""
        client = TRELLIS2APIClient(
            base_url="https://example.com/",
            modal_key="wk-test",
            modal_secret="ws-test",
        )
        assert client.generate_url == "https://example.com/generate"
        assert client.extract_glb_url == "https://example.com/extract_glb"

    def test_init_derives_urls_for_modal_subdomain(self) -> None:
        """Modal subdomain URLs should derive both endpoint URLs."""
        client = TRELLIS2APIClient(
            base_url="https://user--app-generate.modal.run",
            modal_key="wk-test",
            modal_secret="ws-test",
        )
        assert client.generate_url == "https://user--app-generate.modal.run"
        assert client.extract_glb_url == "https://user--app-extract-glb.modal.run"

    def test_last_request_elapsed_is_none_initially(self) -> None:
        """last_request_elapsed should be None before any request."""
        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )
        assert client.last_request_elapsed is None


class TestTRELLIS2APIClientGenerate:
    """Tests for TRELLIS2APIClient.generate()."""

    def test_generate_sends_correct_request(self, tmp_path: Path) -> None:
        """generate() should send POST with correct headers and payload."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test123",
            modal_secret="ws-secret456",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            client.generate(
                image_path=str(image_path),
                seed=42,
                pipeline_type="1024_cascade",
                ss_sampling_steps=12,
                ss_guidance_strength=7.5,
                shape_slat_sampling_steps=10,
                shape_slat_guidance_strength=6.0,
                tex_slat_sampling_steps=8,
                tex_slat_guidance_strength=1.0,
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "https://example.com/generate"

            assert call_args[1]["headers"]["Modal-Key"] == "wk-test123"
            assert call_args[1]["headers"]["Modal-Secret"] == "ws-secret456"
            assert call_args[1]["headers"]["Content-Type"] == "application/json"

            payload = call_args[1]["json"]
            assert payload["seed"] == 42
            assert payload["pipeline_type"] == "1024_cascade"
            assert payload["ss_sampling_steps"] == 12
            assert payload["ss_guidance_strength"] == 7.5
            assert payload["shape_slat_sampling_steps"] == 10
            assert payload["shape_slat_guidance_strength"] == 6.0
            assert payload["tex_slat_sampling_steps"] == 8
            assert payload["tex_slat_guidance_strength"] == 1.0
            assert payload["image"] == base64.b64encode(b"fake png data").decode()

    def test_generate_returns_result_dict(self, tmp_path: Path) -> None:
        """generate() should return dict with state and video."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            result = client.generate(image_path=str(image_path))

            assert "state" in result
            assert "video" in result
            assert result["state"] == "c3RhdGU="
            assert result["video"] == "dmlkZW8="

    def test_generate_uses_default_params(self, tmp_path: Path) -> None:
        """generate() should use defaults when params not specified."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            client.generate(image_path=str(image_path))

            payload = mock_request.call_args[1]["json"]
            assert payload["seed"] == 42
            assert payload["pipeline_type"] == "1024_cascade"
            assert payload["ss_sampling_steps"] == 12
            assert payload["ss_guidance_strength"] == 7.5
            assert payload["shape_slat_sampling_steps"] == 12
            assert payload["shape_slat_guidance_strength"] == 7.5
            assert payload["tex_slat_sampling_steps"] == 12
            assert payload["tex_slat_guidance_strength"] == 1.0

    def test_generate_raises_file_not_found(self) -> None:
        """generate() should raise FileNotFoundError for missing image."""
        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with pytest.raises(FileNotFoundError, match="Image not found"):
            client.generate(image_path="/nonexistent/image.png")

    def test_generate_raises_api_error(self, tmp_path: Path) -> None:
        """generate() should raise APIError on server error response."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=400,
                json=lambda: {
                    "error": {"code": "validation_error", "message": "Invalid image"}
                },
            )

            with pytest.raises(APIError) as exc_info:
                client.generate(image_path=str(image_path))

            assert exc_info.value.code == "validation_error"
            assert exc_info.value.message == "Invalid image"

    @pytest.mark.parametrize(
        "pipeline_type",
        ["512", "1024", "1024_cascade", "1536_cascade"],
    )
    def test_generate_accepts_all_pipeline_types(
        self, tmp_path: Path, pipeline_type: str
    ) -> None:
        """generate() should accept all valid pipeline types."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            client.generate(image_path=str(image_path), pipeline_type=pipeline_type)

            payload = mock_request.call_args[1]["json"]
            assert payload["pipeline_type"] == pipeline_type


class TestTRELLIS2APIClientExtractGLB:
    """Tests for TRELLIS2APIClient.extract_glb()."""

    def test_extract_glb_sends_correct_request(self, tmp_path: Path) -> None:
        """extract_glb() should send POST with correct payload."""
        output_path = tmp_path / "output.glb"

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"glb": base64.b64encode(b"glb data").decode()},
            )

            client.extract_glb(
                state="c3RhdGU=",
                output_path=str(output_path),
                decimation_target=500000,
                texture_size=2048,
                remesh=False,
                remesh_band=0.5,
                remesh_project=0.1,
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "https://example.com/extract_glb"

            payload = call_args[1]["json"]
            assert payload["state"] == "c3RhdGU="
            assert payload["decimation_target"] == 500000
            assert payload["texture_size"] == 2048
            assert payload["remesh"] is False
            assert payload["remesh_band"] == 0.5
            assert payload["remesh_project"] == 0.1

    def test_extract_glb_writes_file(self, tmp_path: Path) -> None:
        """extract_glb() should write GLB data to output path."""
        output_path = tmp_path / "output.glb"

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        glb_data = b"fake glb binary data"
        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"glb": base64.b64encode(glb_data).decode()},
            )

            result = client.extract_glb(
                state="c3RhdGU=",
                output_path=str(output_path),
            )

            assert result == str(output_path)
            assert output_path.exists()
            assert output_path.read_bytes() == glb_data

    def test_extract_glb_uses_default_params(self, tmp_path: Path) -> None:
        """extract_glb() should use defaults when params not specified."""
        output_path = tmp_path / "output.glb"

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"glb": base64.b64encode(b"data").decode()},
            )

            client.extract_glb(state="c3RhdGU=", output_path=str(output_path))

            payload = mock_request.call_args[1]["json"]
            assert payload["decimation_target"] == 1000000
            assert payload["texture_size"] == 4096
            assert payload["remesh"] is True
            assert payload["remesh_band"] == 1.0
            assert payload["remesh_project"] == 0.0

    def test_extract_glb_raises_api_error(self, tmp_path: Path) -> None:
        """extract_glb() should raise APIError on server error."""
        output_path = tmp_path / "output.glb"

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=400,
                json=lambda: {
                    "error": {"code": "cuda_oom", "message": "Out of memory"}
                },
            )

            with pytest.raises(APIError) as exc_info:
                client.extract_glb(state="c3RhdGU=", output_path=str(output_path))

            assert exc_info.value.code == "cuda_oom"


class TestColdStartDetection:
    """Tests for cold start detection."""

    def test_was_cold_start_true_for_slow_request(self, tmp_path: Path) -> None:
        """was_cold_start() returns True if request exceeded threshold."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )
            with patch("time.perf_counter", side_effect=[0.0, 35.0]):
                client.generate(image_path=str(image_path))

        assert client.was_cold_start() is True

    def test_was_cold_start_false_for_fast_request(self, tmp_path: Path) -> None:
        """was_cold_start() returns False if request was fast."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )
            with patch("time.perf_counter", side_effect=[0.0, 5.0]):
                client.generate(image_path=str(image_path))

        assert client.was_cold_start() is False

    def test_was_cold_start_false_when_no_request_made(self) -> None:
        """was_cold_start() returns False if no request made yet."""
        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )
        assert client.was_cold_start() is False

    def test_was_cold_start_custom_threshold(self, tmp_path: Path) -> None:
        """was_cold_start() respects custom threshold."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )
            with patch("time.perf_counter", side_effect=[0.0, 10.0]):
                client.generate(image_path=str(image_path))

        assert client.was_cold_start(threshold=5.0) is True
        assert client.was_cold_start(threshold=15.0) is False


class TestRetryLogic:
    """Tests for request retry logic."""

    def test_retries_on_connection_error(self, tmp_path: Path) -> None:
        """Should retry on ConnectionError."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.side_effect = [
                requests.exceptions.ConnectionError(),
                MagicMock(
                    status_code=200,
                    json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
                ),
            ]
            with patch("time.sleep"):
                result = client.generate(image_path=str(image_path))

        assert result["state"] == "c3RhdGU="
        assert mock_request.call_count == 2

    def test_retries_on_timeout(self, tmp_path: Path) -> None:
        """Should retry on Timeout."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.side_effect = [
                requests.exceptions.Timeout(),
                MagicMock(
                    status_code=200,
                    json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
                ),
            ]
            with patch("time.sleep"):
                result = client.generate(image_path=str(image_path))

        assert result["state"] == "c3RhdGU="
        assert mock_request.call_count == 2

    def test_raises_after_max_retries(self, tmp_path: Path) -> None:
        """Should raise after exhausting retries."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.side_effect = requests.exceptions.ConnectionError()
            with patch("time.sleep"):
                with pytest.raises(requests.exceptions.ConnectionError):
                    client.generate(image_path=str(image_path))

        assert mock_request.call_count == 3  # Initial + 2 retries


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_returns_true_on_success(self) -> None:
        """health_check() returns True when service responds."""
        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=200)
            assert client.health_check() is True

    def test_health_check_returns_false_on_connection_error(self) -> None:
        """health_check() returns False on connection error."""
        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.side_effect = requests.exceptions.ConnectionError()
            assert client.health_check() is False

    def test_health_check_returns_false_on_timeout(self) -> None:
        """health_check() returns False on timeout."""
        client = TRELLIS2APIClient(
            base_url="https://example.com",
            modal_key="wk-test",
            modal_secret="ws-test",
        )

        with patch("requests.request") as mock_request:
            mock_request.side_effect = requests.exceptions.Timeout()
            assert client.health_check() is False


class TestAPIError:
    """Tests for APIError exception."""

    def test_api_error_stores_code_and_message(self) -> None:
        """APIError should store code and message."""
        error = APIError("test_code", "test message")
        assert error.code == "test_code"
        assert error.message == "test message"

    def test_api_error_str_includes_code_and_message(self) -> None:
        """APIError str should include code and message."""
        error = APIError("test_code", "test message")
        assert str(error) == "test_code: test message"
