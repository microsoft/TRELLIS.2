"""
Tests for state serialization.

These tests verify pack_state and unpack_state work correctly
without requiring GPU or the full TRELLIS.2 pipeline.
"""

import json

import numpy as np
import pytest


class MockMeshWithVoxel:
    """Mock MeshWithVoxel for testing without GPU dependencies."""

    def __init__(
        self,
        vertices,
        faces,
        origin,
        voxel_size,
        coords,
        attrs,
        voxel_shape,
        layout,
    ):
        self.vertices = vertices
        self.faces = faces
        self.origin = origin
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout


class MockTensor:
    """Mock tensor that mimics torch.Tensor for CPU operations."""

    def __init__(self, data, device="cpu"):
        self._data = np.array(data)
        self._device = device

    def cpu(self):
        return MockTensor(self._data, device="cpu")

    def numpy(self):
        return self._data

    def tolist(self):
        return self._data.tolist()


@pytest.fixture
def mock_mesh():
    """Create a mock MeshWithVoxel for testing."""
    return MockMeshWithVoxel(
        vertices=MockTensor(np.random.randn(100, 3).astype(np.float32)),
        faces=MockTensor(np.random.randint(0, 100, (50, 3)).astype(np.int32)),
        origin=MockTensor([-0.5, -0.5, -0.5]),
        voxel_size=1 / 512,
        coords=MockTensor(np.random.randint(0, 512, (200, 3))),
        attrs=MockTensor(np.random.randn(200, 6).astype(np.float32)),
        voxel_shape=(1, 6, 512, 512, 512),
        layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
    )


def test_pack_state_contains_all_required_fields(mock_mesh):
    """Verify packed state has all MeshWithVoxel fields."""
    from trellis2_modal.service.state import pack_state

    state = pack_state(mock_mesh)

    required_fields = [
        "vertices",
        "faces",
        "attrs",
        "coords",
        "voxel_size",
        "voxel_shape",
        "origin",
        "layout",
    ]
    for field in required_fields:
        assert field in state, f"Missing field: {field}"


def test_pack_state_converts_tensors_to_numpy(mock_mesh):
    """Verify tensors become numpy arrays."""
    from trellis2_modal.service.state import pack_state

    state = pack_state(mock_mesh)

    assert isinstance(state["vertices"], np.ndarray)
    assert isinstance(state["faces"], np.ndarray)
    assert isinstance(state["attrs"], np.ndarray)
    assert isinstance(state["coords"], np.ndarray)


def test_pack_state_layout_is_json_serializable(mock_mesh):
    """Verify layout uses lists not slice objects."""
    from trellis2_modal.service.state import pack_state

    state = pack_state(mock_mesh)

    # Should not raise - layout must be JSON serializable
    json.dumps(state["layout"])

    # Verify structure
    assert state["layout"]["base_color"] == [0, 3]
    assert state["layout"]["metallic"] == [3, 4]
    assert state["layout"]["roughness"] == [4, 5]
    assert state["layout"]["alpha"] == [5, 6]


def test_pack_state_preserves_shapes(mock_mesh):
    """Verify array shapes are preserved."""
    from trellis2_modal.service.state import pack_state

    state = pack_state(mock_mesh)

    assert state["vertices"].shape == (100, 3)
    assert state["faces"].shape == (50, 3)
    assert state["attrs"].shape == (200, 6)
    assert state["coords"].shape == (200, 3)


def test_pack_state_preserves_scalar_values(mock_mesh):
    """Verify scalar values are preserved."""
    from trellis2_modal.service.state import pack_state

    state = pack_state(mock_mesh)

    assert state["voxel_size"] == pytest.approx(1 / 512)
    assert state["voxel_shape"] == [1, 6, 512, 512, 512]
    assert state["origin"] == [-0.5, -0.5, -0.5]
