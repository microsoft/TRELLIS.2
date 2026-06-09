"""
Tests for TRELLIS2Generator.

These tests use mocks to verify generator behavior without requiring
GPU or the full TRELLIS.2 pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest


class MockPipeline:
    """Mock pipeline for testing."""

    def __init__(self):
        self.models = {"model1": MagicMock()}
        self._device = "cpu"
        # Use MagicMock for methods we want to verify
        self.preprocess_image = MagicMock(side_effect=lambda x: x)
        mock_mesh = MagicMock()
        mock_mesh.simplify = MagicMock()
        self.run = MagicMock(return_value=[mock_mesh])

    def to(self, device):
        self._device = str(device)

    def cuda(self):
        self._device = "cuda"


@pytest.fixture
def mock_pipeline_factory():
    """Factory that returns a mock pipeline."""
    return lambda model_name: MockPipeline()


@pytest.fixture
def generator(mock_pipeline_factory):
    """Create a generator with mock pipeline."""
    from trellis2_modal.service.generator import TRELLIS2Generator

    return TRELLIS2Generator(pipeline_factory=mock_pipeline_factory)


def test_generator_init_state():
    """Test initial state of generator."""
    from trellis2_modal.service.generator import TRELLIS2Generator

    gen = TRELLIS2Generator()
    assert gen.pipeline is None
    assert gen.envmap is None
    assert gen.load_time == 0.0
    assert gen.is_ready is False


def test_generator_is_ready_property(generator):
    """Test is_ready property requires both pipeline and envmap."""
    assert generator.is_ready is False

    # Manually set pipeline (simulating partial load)
    generator.pipeline = MockPipeline()
    assert generator.is_ready is False  # envmap not set

    generator.envmap = "mock_envmap"
    assert generator.is_ready is True


def test_generator_generate_requires_loaded_pipeline(generator):
    """Test that generate fails without loaded pipeline."""
    mock_image = MagicMock()
    with pytest.raises(RuntimeError, match="Pipeline not loaded"):
        generator.generate(mock_image)


def test_generator_generate_calls_pipeline(generator):
    """Test that generate calls the pipeline correctly."""
    # Manually set up pipeline for testing
    generator.pipeline = MockPipeline()

    with patch("trellis2_modal.service.state.pack_state") as mock_pack:
        mock_pack.return_value = {"test": "state"}
        mock_image = MagicMock()

        generator.generate(mock_image, seed=123)

        # Verify pipeline methods were called
        generator.pipeline.preprocess_image.assert_called_once_with(mock_image)
        generator.pipeline.run.assert_called_once()

        # Verify pack_state was called
        mock_pack.assert_called_once()


def test_generator_generate_passes_correct_params(generator):
    """Test that generate passes parameters correctly to pipeline."""
    generator.pipeline = MockPipeline()

    with patch("trellis2_modal.service.state.pack_state") as mock_pack:
        mock_pack.return_value = {}
        mock_image = MagicMock()

        generator.generate(
            mock_image,
            seed=42,
            pipeline_type="512",
            ss_params={"steps": 8},
            shape_params={"steps": 10},
            tex_params={"steps": 12},
            max_num_tokens=10000,
        )

        call_kwargs = generator.pipeline.run.call_args.kwargs
        assert call_kwargs["seed"] == 42
        assert call_kwargs["pipeline_type"] == "512"
        assert call_kwargs["sparse_structure_sampler_params"] == {"steps": 8}
        assert call_kwargs["shape_slat_sampler_params"] == {"steps": 10}
        assert call_kwargs["tex_slat_sampler_params"] == {"steps": 12}
        assert call_kwargs["max_num_tokens"] == 10000


@pytest.mark.parametrize(
    "pipeline_type",
    ["512", "1024", "1024_cascade", "1536_cascade"],
)
def test_generator_accepts_all_pipeline_types(generator, pipeline_type):
    """Test all 4 pipeline types are accepted."""
    generator.pipeline = MockPipeline()

    with patch("trellis2_modal.service.state.pack_state") as mock_pack:
        mock_pack.return_value = {}
        mock_image = MagicMock()

        # Should not raise
        generator.generate(mock_image, pipeline_type=pipeline_type)

        call_kwargs = generator.pipeline.run.call_args.kwargs
        assert call_kwargs["pipeline_type"] == pipeline_type


def test_generator_render_preview_requires_ready(generator):
    """Test that render_preview_video fails without full initialization."""
    with pytest.raises(RuntimeError, match="Generator not ready"):
        generator.render_preview_video({})


def test_generator_extract_glb_requires_ready(generator):
    """Test that extract_glb fails without full initialization."""
    with pytest.raises(RuntimeError, match="Generator not ready"):
        generator.extract_glb({})


def test_generator_load_time_initialized():
    """Test load_time starts at 0."""
    from trellis2_modal.service.generator import TRELLIS2Generator

    gen = TRELLIS2Generator()
    assert gen.load_time == 0.0
