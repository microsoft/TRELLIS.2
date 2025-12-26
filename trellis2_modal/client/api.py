"""
API client for the Modal TRELLIS.2 service.

Credentials can be provided via:
1. Constructor parameters: modal_key and modal_secret
2. Environment variables: TRELLIS2_MODAL_KEY and TRELLIS2_MODAL_SECRET
3. Secrets file: ~/.trellis2_modal_secrets.json
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


# Default path for local secrets file
SECRETS_FILE_PATH = Path.home() / ".trellis2_modal_secrets.json"


def load_credentials(
    modal_key: str | None = None,
    modal_secret: str | None = None,
    secrets_file: Path | None = None,
) -> tuple[str, str]:
    """
    Load Modal Proxy Auth credentials from various sources.

    Priority order:
    1. Explicit parameters (modal_key, modal_secret)
    2. Environment variables (TRELLIS2_MODAL_KEY, TRELLIS2_MODAL_SECRET)
    3. Local secrets file (~/.trellis2_modal_secrets.json)

    Args:
        modal_key: Optional explicit Modal-Key value
        modal_secret: Optional explicit Modal-Secret value
        secrets_file: Optional path to secrets file (default: ~/.trellis2_modal_secrets.json)

    Returns:
        Tuple of (modal_key, modal_secret)

    Raises:
        ValueError: If credentials cannot be found from any source
    """
    # 1. Check explicit parameters
    if modal_key and modal_secret:
        return modal_key, modal_secret

    # 2. Check environment variables
    env_key = os.environ.get("TRELLIS2_MODAL_KEY")
    env_secret = os.environ.get("TRELLIS2_MODAL_SECRET")
    if env_key and env_secret:
        return env_key, env_secret

    # 3. Check local secrets file
    secrets_path = secrets_file or SECRETS_FILE_PATH
    if secrets_path.exists():
        try:
            data = json.loads(secrets_path.read_text())
            file_key = data.get("modal_key")
            file_secret = data.get("modal_secret")
            if file_key and file_secret:
                return file_key, file_secret
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid secrets file format: {e}") from e

    raise ValueError(
        "Modal credentials not found. Provide credentials via:\n"
        "  1. Constructor parameters: modal_key and modal_secret\n"
        "  2. Environment variables: TRELLIS2_MODAL_KEY and TRELLIS2_MODAL_SECRET\n"
        "  3. Secrets file: ~/.trellis2_modal_secrets.json with keys 'modal_key' and 'modal_secret'\n"
        "\n"
        "Create Proxy Auth Tokens in the Modal dashboard at /settings/proxy-auth-tokens"
    )


class TRELLIS2APIClient:
    """Client for the Modal-deployed TRELLIS.2 service."""

    # Default timeout for requests (10 minutes for cascade pipelines)
    DEFAULT_TIMEOUT = 600

    # Default cold start threshold in seconds
    DEFAULT_COLD_START_THRESHOLD = 30.0

    # Retry configuration
    MAX_RETRIES = 2
    INITIAL_BACKOFF = 1.0

    def __init__(
        self,
        base_url: str,
        modal_key: str | None = None,
        modal_secret: str | None = None,
    ) -> None:
        """
        Initialize the API client.

        Args:
            base_url: Modal endpoint URL. For Modal subdomain routing, provide
                the generate endpoint URL (e.g., https://...generate.modal.run).
                The extract_glb URL will be derived automatically.
            modal_key: Modal Proxy Auth key (or use env/secrets file)
            modal_secret: Modal Proxy Auth secret (or use env/secrets file)

        Raises:
            ValueError: If credentials cannot be loaded from any source
        """
        base_url = base_url.rstrip("/")

        # Detect Modal subdomain routing pattern and derive endpoint URLs
        if "-generate.modal.run" in base_url:
            # Modal subdomain pattern: derive extract_glb URL
            self.generate_url = base_url
            self.extract_glb_url = base_url.replace("-generate.modal.run", "-extract-glb.modal.run")
        elif "-extract-glb.modal.run" in base_url:
            # User provided extract_glb URL, derive generate URL
            self.extract_glb_url = base_url
            self.generate_url = base_url.replace("-extract-glb.modal.run", "-generate.modal.run")
        else:
            # Path-based routing: append /generate and /extract_glb
            self.generate_url = f"{base_url}/generate"
            self.extract_glb_url = f"{base_url}/extract_glb"

        self.modal_key, self.modal_secret = load_credentials(modal_key, modal_secret)
        self._last_request_elapsed: float | None = None

    @property
    def last_request_elapsed(self) -> float | None:
        """Return elapsed time of last request in seconds, or None if no request made."""
        return self._last_request_elapsed

    def was_cold_start(self, threshold: float | None = None) -> bool:
        """Check if the last request was likely a cold start (>threshold seconds)."""
        if self._last_request_elapsed is None:
            return False
        if threshold is None:
            threshold = self.DEFAULT_COLD_START_THRESHOLD
        return self._last_request_elapsed > threshold

    def _headers(self) -> dict[str, str]:
        """Return headers for API requests with Modal Proxy Auth."""
        return {
            "Modal-Key": self.modal_key,
            "Modal-Secret": self.modal_secret,
            "Content-Type": "application/json",
        }

    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """
        Make HTTP request with retry logic for transient failures.

        Retries on ConnectionError and Timeout with exponential backoff.
        """
        last_exception = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return requests.request(method, url, **kwargs)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                last_exception = e
                if attempt < self.MAX_RETRIES:
                    delay = self.INITIAL_BACKOFF * (2**attempt)
                    time.sleep(delay)

        raise last_exception  # type: ignore[misc]

    def _check_error(self, response_data: dict[str, Any]) -> None:
        """Check response for errors and raise APIError if found."""
        if "error" in response_data:
            error = response_data["error"]
            raise APIError(error["code"], error["message"])

    def generate(
        self,
        image_path: str,
        seed: int = 42,
        pipeline_type: str = "1024_cascade",
        ss_sampling_steps: int = 12,
        ss_guidance_strength: float = 7.5,
        shape_slat_sampling_steps: int = 12,
        shape_slat_guidance_strength: float = 7.5,
        tex_slat_sampling_steps: int = 12,
        tex_slat_guidance_strength: float = 1.0,
    ) -> dict[str, Any]:
        """
        Generate 3D from image via the Modal service.

        Args:
            image_path: Path to input image
            seed: Random seed for reproducibility
            pipeline_type: One of "512", "1024", "1024_cascade", "1536_cascade"
            ss_sampling_steps: Sparse structure sampling steps
            ss_guidance_strength: Sparse structure guidance strength
            shape_slat_sampling_steps: Shape SLAT sampling steps
            shape_slat_guidance_strength: Shape SLAT guidance strength
            tex_slat_sampling_steps: Texture SLAT sampling steps
            tex_slat_guidance_strength: Texture SLAT guidance strength

        Returns:
            Dict with 'state' (base64 compressed) and 'video' (base64)

        Raises:
            APIError: If the request fails
            FileNotFoundError: If image_path doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")

        payload = {
            "image": image_b64,
            "seed": seed,
            "pipeline_type": pipeline_type,
            "ss_sampling_steps": ss_sampling_steps,
            "ss_guidance_strength": ss_guidance_strength,
            "shape_slat_sampling_steps": shape_slat_sampling_steps,
            "shape_slat_guidance_strength": shape_slat_guidance_strength,
            "tex_slat_sampling_steps": tex_slat_sampling_steps,
            "tex_slat_guidance_strength": tex_slat_guidance_strength,
        }

        start = time.perf_counter()
        response = self._request_with_retry(
            "POST",
            self.generate_url,
            headers=self._headers(),
            json=payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        self._last_request_elapsed = time.perf_counter() - start

        result = response.json()
        self._check_error(result)

        return result

    def extract_glb(
        self,
        state: str,
        output_path: str,
        decimation_target: int = 1000000,
        texture_size: int = 4096,
        remesh: bool = True,
        remesh_band: float = 1.0,
        remesh_project: float = 0.0,
    ) -> str:
        """
        Extract GLB mesh from generation state.

        Args:
            state: Base64 compressed state string from generate()
            output_path: Path to write GLB file
            decimation_target: Target vertex count (default 1M)
            texture_size: Texture resolution (512, 1024, 2048, 4096)
            remesh: Whether to remesh for cleaner topology
            remesh_band: Remesh band size
            remesh_project: Remesh projection factor

        Returns:
            Path to written GLB file

        Raises:
            APIError: If the request fails
        """
        payload = {
            "state": state,
            "decimation_target": decimation_target,
            "texture_size": texture_size,
            "remesh": remesh,
            "remesh_band": remesh_band,
            "remesh_project": remesh_project,
        }

        start = time.perf_counter()
        response = self._request_with_retry(
            "POST",
            self.extract_glb_url,
            headers=self._headers(),
            json=payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        self._last_request_elapsed = time.perf_counter() - start

        result = response.json()
        self._check_error(result)

        glb_bytes = base64.b64decode(result["glb"])
        Path(output_path).write_bytes(glb_bytes)

        return output_path

    def health_check(self) -> bool:
        """
        Check if the generate endpoint is reachable.

        Returns:
            True if endpoint is reachable, False otherwise
        """
        try:
            requests.request(
                "HEAD",
                self.generate_url,
                headers=self._headers(),
                timeout=10,
            )
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False


class APIError(Exception):
    """Exception raised for API errors."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")
