"""
State compression utilities for TRELLIS.2 client-server communication.

Handles LZ4 compression of MeshWithVoxel state for efficient transfer
between server and client. The compressed state is stored client-side
in gr.State to maintain stateless server design.

Security note: Uses pickle for serialization. Only decompress data from
trusted sources (our own server). Data flow is: server creates state ->
compress -> client stores -> client sends back -> server decompresses.
"""

from __future__ import annotations

import pickle
from typing import Any

import lz4.frame


def compress_state(state: dict[str, Any]) -> bytes:
    """
    Compress generation state for storage/transfer.

    Uses pickle serialization + LZ4 frame compression.
    LZ4 frame format is streaming-friendly and self-delimiting.

    Args:
        state: Dictionary containing numpy arrays and metadata

    Returns:
        LZ4-compressed bytes
    """
    serialized = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    return lz4.frame.compress(serialized)


def decompress_state(data: bytes) -> dict[str, Any]:
    """
    Decompress generation state.

    Args:
        data: LZ4-compressed bytes from compress_state

    Returns:
        Original state dictionary with numpy arrays
    """
    decompressed = lz4.frame.decompress(data)
    return pickle.loads(decompressed)


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 for API transmission.

    Args:
        image_path: Path to image file

    Returns:
        Base64-encoded image string
    """
    raise NotImplementedError("Image encoding not yet implemented")


def decode_file(data: str, output_path: str) -> None:
    """
    Decode base64 file data and write to disk.

    Args:
        data: Base64-encoded file contents
        output_path: Path to write decoded file
    """
    raise NotImplementedError("File decoding not yet implemented")
