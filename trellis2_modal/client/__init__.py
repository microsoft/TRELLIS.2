"""
TRELLIS.2 Modal Client Package.

Provides API client and Gradio UI for interacting with the Modal service.

Usage:
    # As library
    from trellis2_modal.client import TRELLIS2APIClient, APIError

    # As application
    python -m trellis2_modal.client.app
"""

from .api import APIError, TRELLIS2APIClient
from .compression import compress_state, decompress_state

__all__ = [
    "TRELLIS2APIClient",
    "APIError",
    "compress_state",
    "decompress_state",
]
