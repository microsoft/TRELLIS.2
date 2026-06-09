"""
Tests for LZ4 compression utilities.

Tests the compress_state/decompress_state functions that handle
serialization and compression of TRELLIS generation state.
"""

import pickle

import numpy as np

from trellis2_modal.client.compression import compress_state, decompress_state


class TestCompressState:
    """Tests for compress_state function."""

    def test_returns_bytes(self) -> None:
        """Compress should return bytes."""
        state = {"key": "value"}
        result = compress_state(state)
        assert isinstance(result, bytes)

    def test_returns_non_empty_bytes(self) -> None:
        """Compressed result should not be empty."""
        state = {"key": "value"}
        result = compress_state(state)
        assert len(result) > 0

    def test_handles_nested_dicts(self) -> None:
        """Should compress nested dictionary structures."""
        state = {"outer": {"inner": {"deep": 42}}}
        result = compress_state(state)
        assert isinstance(result, bytes)


class TestDecompressState:
    """Tests for decompress_state function."""

    def test_returns_dict(self) -> None:
        """Decompress should return a dictionary."""
        state = {"key": "value"}
        compressed = compress_state(state)
        result = decompress_state(compressed)
        assert isinstance(result, dict)


class TestCompressionRoundtrip:
    """Tests for compress/decompress round-trip."""

    def test_roundtrip_simple_dict(self) -> None:
        """Simple dict should survive round-trip."""
        state = {"key": "value", "number": 42}
        compressed = compress_state(state)
        decompressed = decompress_state(compressed)
        assert decompressed == state

    def test_roundtrip_with_list(self) -> None:
        """List values should survive round-trip."""
        state = {"aabb": [0, 0, 0, 1, 1, 1]}
        compressed = compress_state(state)
        decompressed = decompress_state(compressed)
        assert decompressed == state

    def test_roundtrip_numpy_array(self) -> None:
        """Numpy array should survive round-trip."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        state = {"data": original}
        compressed = compress_state(state)
        decompressed = decompress_state(compressed)
        np.testing.assert_array_equal(decompressed["data"], original)

    def test_roundtrip_preserves_dtype(self) -> None:
        """Numpy array dtype should be preserved."""
        original = np.array([1, 2, 3], dtype=np.int64)
        state = {"data": original}
        compressed = compress_state(state)
        decompressed = decompress_state(compressed)
        assert decompressed["data"].dtype == np.int64

    def test_roundtrip_multidimensional_array(self) -> None:
        """Multi-dimensional arrays should survive round-trip."""
        original = np.random.randn(100, 3).astype(np.float32)
        state = {"positions": original}
        compressed = compress_state(state)
        decompressed = decompress_state(compressed)
        np.testing.assert_array_equal(decompressed["positions"], original)

    def test_roundtrip_realistic_state(self) -> None:
        """Realistic state structure (from pack_state) should survive."""
        state = {
            "gaussian": {
                "aabb": [0, 0, 0, 1, 1, 1],
                "sh_degree": 0,
                "scaling_activation": "exp",
                "_xyz": np.random.randn(1000, 3).astype(np.float32),
                "_features_dc": np.random.randn(1000, 3).astype(np.float32),
                "_scaling": np.random.randn(1000, 3).astype(np.float32),
                "_rotation": np.random.randn(1000, 4).astype(np.float32),
                "_opacity": np.random.randn(1000, 1).astype(np.float32),
            },
            "mesh": {
                "vertices": np.random.randn(500, 3).astype(np.float32),
                "faces": np.arange(1500).reshape(500, 3).astype(np.int64),
            },
        }
        compressed = compress_state(state)
        decompressed = decompress_state(compressed)

        # Check scalar values
        assert decompressed["gaussian"]["aabb"] == state["gaussian"]["aabb"]
        assert decompressed["gaussian"]["sh_degree"] == state["gaussian"]["sh_degree"]

        # Check numpy arrays
        np.testing.assert_array_equal(
            decompressed["gaussian"]["_xyz"], state["gaussian"]["_xyz"]
        )
        np.testing.assert_array_equal(
            decompressed["mesh"]["faces"], state["mesh"]["faces"]
        )

    def test_roundtrip_empty_dict(self) -> None:
        """Empty dict should survive round-trip."""
        state: dict = {}
        compressed = compress_state(state)
        decompressed = decompress_state(compressed)
        assert decompressed == state


class TestCompressionEfficiency:
    """Tests for compression efficiency."""

    def test_compresses_repetitive_data(self) -> None:
        """Repetitive data should compress well."""
        # Zeros compress very well
        state = {"zeros": np.zeros((1000, 3), dtype=np.float32)}
        serialized = pickle.dumps(state)
        compressed = compress_state(state)
        assert len(compressed) < len(serialized)

    def test_compresses_realistic_state(self) -> None:
        """Realistic state should have meaningful compression."""
        state = {
            "gaussian": {
                "_xyz": np.random.randn(10000, 3).astype(np.float32),
                "_features_dc": np.random.randn(10000, 3).astype(np.float32),
            }
        }
        serialized = pickle.dumps(state)
        compressed = compress_state(state)
        # LZ4 on random floats won't compress much, but should not expand significantly
        # The overhead should be minimal (LZ4 frame header is small)
        assert len(compressed) < len(serialized) * 1.1
