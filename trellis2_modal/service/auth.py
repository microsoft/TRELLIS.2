"""
API key authentication for the Modal TRELLIS.2 service.

Handles validation of API keys stored in a Modal Volume. Keys are
stored in JSON format with metadata including creation date, quotas,
and usage tracking.
"""

from __future__ import annotations

from typing import Any


def validate_api_key(
    key: str | None,
    keys_data: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    """
    Validate an API key against the provided keys data.

    Pure function that checks format and looks up the key in the provided
    dictionary. Does not perform I/O - caller is responsible for loading
    keys_data.

    Args:
        key: API key string (format: sk_xxx_...) or None
        keys_data: Dictionary with 'keys' field containing key->info mappings

    Returns:
        Tuple of (is_valid, key_info) where:
        - is_valid: True if key is valid and active
        - key_info: Dict with key metadata if valid, None otherwise
    """
    if not key:
        return False, None

    if not key.startswith("sk_"):
        return False, None

    keys = keys_data.get("keys", {})
    if key not in keys:
        return False, None

    key_info = keys[key]
    if not key_info.get("active", True):
        return False, None

    return True, key_info


def load_keys(path: str) -> dict[str, Any]:
    """
    Load API keys from a JSON file.

    If the file doesn't exist, returns an empty keys structure.
    This simplifies initial deployment - no need to pre-create keys.json.

    Args:
        path: Path to the keys.json file

    Returns:
        Dictionary with 'version' and 'keys' fields

    Raises:
        ValueError: If file exists but contains invalid JSON
    """
    import json
    from pathlib import Path

    file_path = Path(path)
    if not file_path.exists():
        return {"version": 1, "keys": {}}

    try:
        return json.loads(file_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def save_keys(path: str, data: dict[str, Any]) -> None:
    """
    Save API keys to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        path: Path to the keys.json file
        data: Dictionary to save (should have 'version' and 'keys' fields)
    """
    import json
    from pathlib import Path

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=2))


def check_auth(
    api_key: str | None,
    keys_path: str = "/data/keys.json",
) -> tuple[bool, dict[str, Any] | None]:
    """
    Check API key authentication.

    Loads keys from file and validates the provided key.
    Convenience function that combines load_keys and validate_api_key.

    Args:
        api_key: API key from request header (may be None)
        keys_path: Path to keys.json file

    Returns:
        Tuple of (is_valid, key_info) where:
        - is_valid: True if key is valid and active
        - key_info: Dict with key metadata if valid, None otherwise
    """
    keys_data = load_keys(keys_path)
    return validate_api_key(api_key, keys_data)


def mask_api_key(key: str | None) -> str:
    """
    Mask an API key for safe logging.

    Shows first 7 chars (sk_xxx_) and last 4 chars, masks the middle.

    Args:
        key: API key to mask

    Returns:
        Masked key like 'sk_dev_...abcd' or '<none>' if None
    """
    if not key:
        return "<none>"
    if len(key) <= 11:
        return key[:7] + "..."
    return key[:7] + "..." + key[-4:]


def increment_usage(key: str, keys_data: dict[str, Any]) -> None:
    """
    Increment the usage counter for an API key.

    Pure function that mutates keys_data in place. Caller is responsible
    for persisting changes with save_keys if needed.

    Silently ignores unknown keys or missing 'keys' field.

    Args:
        key: API key that was used
        keys_data: Dictionary with 'keys' field containing key->info mappings
    """
    from datetime import datetime, timezone

    keys = keys_data.get("keys", {})
    if key not in keys:
        return

    key_info = keys[key]
    key_info["usage_count"] = key_info.get("usage_count", 0) + 1
    key_info["last_used"] = datetime.now(timezone.utc).isoformat()


def check_rate_limit(key_info: dict[str, Any]) -> bool:
    """
    Check if a key is within its rate limit quota.

    Args:
        key_info: Dictionary with key metadata (usage_count, quota)

    Returns:
        True if within quota (or no quota set), False if quota exceeded.
    """
    quota = key_info.get("quota")
    if quota is None:
        return True  # No quota = unlimited

    usage_count = key_info.get("usage_count", 0)
    return usage_count < quota


def generate_api_key(prefix: str = "dev") -> str:
    """
    Generate a new API key.

    Keys are in format: sk_{prefix}_{random_hex}
    Example: sk_dev_a1b2c3d4e5f6g7h8

    Args:
        prefix: Key prefix, typically "dev" or "live"

    Returns:
        New API key string
    """
    import secrets

    random_part = secrets.token_hex(16)  # 32 hex chars
    return f"sk_{prefix}_{random_part}"


def add_key(
    keys_data: dict[str, Any],
    name: str,
    prefix: str = "dev",
    quota: int | None = None,
) -> str:
    """
    Add a new API key to keys_data.

    Args:
        keys_data: Keys data dict to modify
        name: Human-readable name for the key
        prefix: Key prefix, typically "dev" or "live"
        quota: Optional request quota limit

    Returns:
        The generated API key
    """
    from datetime import datetime, timezone

    key = generate_api_key(prefix)

    if "keys" not in keys_data:
        keys_data["keys"] = {}

    key_info: dict[str, Any] = {
        "name": name,
        "active": True,
        "created": datetime.now(timezone.utc).isoformat(),
        "usage_count": 0,
    }
    if quota is not None:
        key_info["quota"] = quota

    keys_data["keys"][key] = key_info
    return key


def revoke_key(keys_data: dict[str, Any], key: str) -> bool:
    """
    Revoke an API key.

    Args:
        keys_data: Keys data dict to modify
        key: API key to revoke

    Returns:
        True if key was found and revoked, False otherwise
    """
    keys = keys_data.get("keys", {})
    if key not in keys:
        return False

    keys[key]["active"] = False
    return True


# CLI interface
def _cli_add_key(args: Any, keys_path: str) -> None:
    """CLI handler for add-key command."""
    keys_data = load_keys(keys_path)
    key = add_key(
        keys_data,
        name=args.name,
        prefix=args.prefix,
        quota=args.quota,
    )
    save_keys(keys_path, keys_data)
    print(f"Created API key: {key}")
    print(f"Name: {args.name}")
    if args.quota:
        print(f"Quota: {args.quota}")


def _cli_list_keys(args: Any, keys_path: str) -> None:
    """CLI handler for list-keys command."""
    keys_data = load_keys(keys_path)
    keys = keys_data.get("keys", {})

    if not keys:
        print("No API keys found.")
        return

    for key, info in keys.items():
        status = "active" if info.get("active", True) else "revoked"
        usage = info.get("usage_count", 0)
        print(
            f"{mask_api_key(key):20} {info.get('name', 'unnamed'):15} {status:8} usage: {usage}"
        )


def _cli_revoke_key(args: Any, keys_path: str) -> None:
    """CLI handler for revoke-key command."""
    keys_data = load_keys(keys_path)
    if revoke_key(keys_data, args.key):
        save_keys(keys_path, keys_data)
        print(f"Revoked key: {mask_api_key(args.key)}")
    else:
        print(f"Key not found: {mask_api_key(args.key)}")


def _cli_usage(args: Any, keys_path: str) -> None:
    """CLI handler for usage command."""
    keys_data = load_keys(keys_path)
    keys = keys_data.get("keys", {})

    if args.key not in keys:
        print(f"Key not found: {mask_api_key(args.key)}")
        return

    info = keys[args.key]
    print(f"Key:         {mask_api_key(args.key)}")
    print(f"Name:        {info.get('name', 'unnamed')}")
    print(f"Status:      {'active' if info.get('active', True) else 'revoked'}")
    print(f"Created:     {info.get('created', 'unknown')}")
    print(f"Usage count: {info.get('usage_count', 0)}")
    print(f"Last used:   {info.get('last_used', 'never')}")
    if "quota" in info:
        print(f"Quota:       {info['quota']}")


def main() -> None:
    """CLI entrypoint for API key management."""
    import argparse

    from .config import API_KEYS_PATH

    parser = argparse.ArgumentParser(
        description="Manage API keys for TRELLIS.2 Modal service"
    )
    parser.add_argument(
        "--keys-file",
        default=API_KEYS_PATH,
        help=f"Path to keys.json file (default: {API_KEYS_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # add-key command
    add_parser = subparsers.add_parser("add-key", help="Add a new API key")
    add_parser.add_argument(
        "--name", required=True, help="Human-readable name for the key"
    )
    add_parser.add_argument("--prefix", default="dev", help="Key prefix (dev or live)")
    add_parser.add_argument("--quota", type=int, help="Optional request quota limit")

    # list-keys command
    subparsers.add_parser("list-keys", help="List all API keys")

    # revoke-key command
    revoke_parser = subparsers.add_parser("revoke-key", help="Revoke an API key")
    revoke_parser.add_argument("key", help="API key to revoke")

    # usage command
    usage_parser = subparsers.add_parser("usage", help="Show usage for an API key")
    usage_parser.add_argument("key", help="API key to show usage for")

    args = parser.parse_args()
    keys_path = args.keys_file

    if args.command == "add-key":
        _cli_add_key(args, keys_path)
    elif args.command == "list-keys":
        _cli_list_keys(args, keys_path)
    elif args.command == "revoke-key":
        _cli_revoke_key(args, keys_path)
    elif args.command == "usage":
        _cli_usage(args, keys_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
