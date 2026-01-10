"""
Tests for API key authentication.

Tests validate_api_key function with various valid and invalid key scenarios.
Tests load_keys and save_keys for file I/O.
"""

import json
from pathlib import Path

import pytest

from trellis2_modal.service.auth import (
    add_key,
    check_auth,
    generate_api_key,
    increment_usage,
    load_keys,
    mask_api_key,
    revoke_key,
    save_keys,
    validate_api_key,
)


class TestValidateApiKeyFormat:
    """Test API key format validation."""

    def test_none_key_returns_invalid(self) -> None:
        """None key should return invalid."""
        keys_data = {"keys": {}}
        is_valid, key_info = validate_api_key(None, keys_data)
        assert is_valid is False
        assert key_info is None

    def test_empty_key_returns_invalid(self) -> None:
        """Empty string key should return invalid."""
        keys_data = {"keys": {}}
        is_valid, key_info = validate_api_key("", keys_data)
        assert is_valid is False
        assert key_info is None

    def test_key_without_prefix_returns_invalid(self) -> None:
        """Key without 'sk_' prefix should return invalid."""
        keys_data = {"keys": {"not_a_valid_key": {"active": True}}}
        is_valid, key_info = validate_api_key("not_a_valid_key", keys_data)
        assert is_valid is False
        assert key_info is None


class TestValidateApiKeyLookup:
    """Test API key lookup in keys data."""

    def test_valid_key_returns_valid_and_info(self) -> None:
        """Valid active key should return valid with key info."""
        keys_data = {
            "keys": {
                "sk_dev_test123": {
                    "name": "test",
                    "active": True,
                    "created": "2025-01-01",
                }
            }
        }
        is_valid, key_info = validate_api_key("sk_dev_test123", keys_data)
        assert is_valid is True
        assert key_info is not None
        assert key_info["name"] == "test"

    def test_unknown_key_returns_invalid(self) -> None:
        """Key not in keys_data should return invalid."""
        keys_data = {"keys": {"sk_dev_other": {"active": True}}}
        is_valid, key_info = validate_api_key("sk_dev_unknown", keys_data)
        assert is_valid is False
        assert key_info is None

    def test_inactive_key_returns_invalid(self) -> None:
        """Inactive key should return invalid."""
        keys_data = {
            "keys": {
                "sk_dev_revoked": {
                    "name": "revoked",
                    "active": False,
                }
            }
        }
        is_valid, key_info = validate_api_key("sk_dev_revoked", keys_data)
        assert is_valid is False
        assert key_info is None

    def test_key_without_active_field_defaults_to_active(self) -> None:
        """Key without 'active' field should default to active."""
        keys_data = {"keys": {"sk_dev_noactive": {"name": "no_active_field"}}}
        is_valid, key_info = validate_api_key("sk_dev_noactive", keys_data)
        assert is_valid is True
        assert key_info is not None


class TestValidateApiKeyEdgeCases:
    """Test edge cases in API key validation."""

    def test_empty_keys_data_returns_invalid(self) -> None:
        """Empty keys dict should return invalid for any key."""
        keys_data = {"keys": {}}
        is_valid, key_info = validate_api_key("sk_dev_test", keys_data)
        assert is_valid is False
        assert key_info is None

    def test_missing_keys_field_returns_invalid(self) -> None:
        """Missing 'keys' field should return invalid."""
        keys_data = {}
        is_valid, key_info = validate_api_key("sk_dev_test", keys_data)
        assert is_valid is False
        assert key_info is None

    def test_live_key_prefix_accepted(self) -> None:
        """Key with 'sk_live_' prefix should be accepted."""
        keys_data = {"keys": {"sk_live_prod123": {"active": True}}}
        is_valid, key_info = validate_api_key("sk_live_prod123", keys_data)
        assert is_valid is True


class TestLoadKeys:
    """Tests for load_keys function."""

    def test_loads_valid_json_file(self, tmp_path: Path) -> None:
        """Valid keys.json should load correctly."""
        keys_file = tmp_path / "keys.json"
        expected = {
            "version": 1,
            "keys": {"sk_dev_test": {"name": "test", "active": True}},
        }
        keys_file.write_text(json.dumps(expected))

        result = load_keys(str(keys_file))
        assert result == expected

    def test_missing_file_returns_empty_structure(self, tmp_path: Path) -> None:
        """Missing file should return empty keys structure."""
        missing_file = tmp_path / "nonexistent.json"

        result = load_keys(str(missing_file))
        assert result == {"version": 1, "keys": {}}

    def test_invalid_json_raises_error(self, tmp_path: Path) -> None:
        """Invalid JSON should raise ValueError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_keys(str(bad_file))


class TestSaveKeys:
    """Tests for save_keys function."""

    def test_saves_keys_to_file(self, tmp_path: Path) -> None:
        """save_keys should write valid JSON to file."""
        keys_file = tmp_path / "keys.json"
        data = {"version": 1, "keys": {"sk_dev_test": {"active": True}}}

        save_keys(str(keys_file), data)

        assert keys_file.exists()
        loaded = json.loads(keys_file.read_text())
        assert loaded == data

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """save_keys should create parent directories if needed."""
        nested_file = tmp_path / "subdir" / "keys.json"
        data = {"version": 1, "keys": {}}

        save_keys(str(nested_file), data)

        assert nested_file.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """save_keys should overwrite existing file."""
        keys_file = tmp_path / "keys.json"
        keys_file.write_text('{"old": "data"}')

        new_data = {"version": 2, "keys": {"sk_new": {}}}
        save_keys(str(keys_file), new_data)

        loaded = json.loads(keys_file.read_text())
        assert loaded == new_data


class TestCheckAuth:
    """Tests for check_auth convenience function."""

    def test_valid_key_with_file(self, tmp_path: Path) -> None:
        """check_auth should validate key from file."""
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "keys": {"sk_dev_test123": {"name": "test", "active": True}},
                }
            )
        )

        is_valid, key_info = check_auth("sk_dev_test123", str(keys_file))
        assert is_valid is True
        assert key_info["name"] == "test"

    def test_invalid_key_with_file(self, tmp_path: Path) -> None:
        """check_auth should reject unknown key."""
        keys_file = tmp_path / "keys.json"
        keys_file.write_text(json.dumps({"version": 1, "keys": {}}))

        is_valid, key_info = check_auth("sk_dev_unknown", str(keys_file))
        assert is_valid is False
        assert key_info is None

    def test_missing_file_rejects_all_keys(self, tmp_path: Path) -> None:
        """check_auth with missing file should reject any key."""
        missing_file = tmp_path / "nonexistent.json"

        is_valid, key_info = check_auth("sk_dev_test", str(missing_file))
        assert is_valid is False
        assert key_info is None


class TestMaskApiKey:
    """Tests for mask_api_key function."""

    def test_masks_long_key(self) -> None:
        """Long key should be masked with ... in middle."""
        result = mask_api_key("sk_dev_abc123def456xyz")
        assert result == "sk_dev_...6xyz"

    def test_masks_short_key(self) -> None:
        """Short key should show first 7 chars + ..."""
        result = mask_api_key("sk_dev_a")
        assert result == "sk_dev_..."

    def test_none_key_returns_placeholder(self) -> None:
        """None key should return '<none>'."""
        result = mask_api_key(None)
        assert result == "<none>"

    def test_empty_key_returns_placeholder(self) -> None:
        """Empty key should return '<none>'."""
        result = mask_api_key("")
        assert result == "<none>"


class TestIncrementUsage:
    """Tests for increment_usage function."""

    def test_increments_usage_count(self) -> None:
        """increment_usage should increase usage_count by 1."""
        keys_data = {
            "keys": {
                "sk_dev_test123": {"name": "test", "active": True, "usage_count": 5}
            }
        }
        increment_usage("sk_dev_test123", keys_data)
        assert keys_data["keys"]["sk_dev_test123"]["usage_count"] == 6

    def test_initializes_usage_count_if_missing(self) -> None:
        """increment_usage should initialize usage_count to 1 if not present."""
        keys_data = {"keys": {"sk_dev_test123": {"name": "test", "active": True}}}
        increment_usage("sk_dev_test123", keys_data)
        assert keys_data["keys"]["sk_dev_test123"]["usage_count"] == 1

    def test_updates_last_used_timestamp(self) -> None:
        """increment_usage should update last_used to ISO timestamp."""
        keys_data = {"keys": {"sk_dev_test123": {"name": "test", "active": True}}}
        increment_usage("sk_dev_test123", keys_data)
        assert "last_used" in keys_data["keys"]["sk_dev_test123"]
        # Verify it's an ISO format timestamp
        last_used = keys_data["keys"]["sk_dev_test123"]["last_used"]
        assert last_used.startswith("20")  # Year starts with 20xx
        assert "T" in last_used  # ISO format has T separator

    def test_does_nothing_for_unknown_key(self) -> None:
        """increment_usage should silently ignore unknown keys."""
        keys_data = {"keys": {"sk_dev_other": {"active": True}}}
        increment_usage("sk_dev_unknown", keys_data)
        # Should not raise, just silently return
        assert "sk_dev_unknown" not in keys_data["keys"]

    def test_does_nothing_for_empty_keys_data(self) -> None:
        """increment_usage should handle missing 'keys' field gracefully."""
        keys_data = {}
        increment_usage("sk_dev_test", keys_data)
        # Should not raise or modify keys_data
        assert "keys" not in keys_data


class TestGenerateApiKey:
    """Tests for generate_api_key function."""

    def test_generates_key_with_dev_prefix(self) -> None:
        """Generated key should start with sk_dev_."""
        key = generate_api_key("dev")
        assert key.startswith("sk_dev_")

    def test_generates_key_with_live_prefix(self) -> None:
        """Generated key should start with sk_live_."""
        key = generate_api_key("live")
        assert key.startswith("sk_live_")

    def test_generates_unique_keys(self) -> None:
        """Each call should generate a unique key."""
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_key_has_correct_length(self) -> None:
        """Generated key should have sk_dev_ + 32 hex chars."""
        key = generate_api_key("dev")
        # sk_dev_ is 7 chars, random part is 32 hex chars
        assert len(key) == 7 + 32


class TestAddKey:
    """Tests for add_key function."""

    def test_adds_key_to_keys_data(self) -> None:
        """add_key should add new key to keys_data."""
        keys_data = {"keys": {}}
        key = add_key(keys_data, name="test")
        assert key in keys_data["keys"]

    def test_sets_key_metadata(self) -> None:
        """add_key should set name, active, created, usage_count."""
        keys_data = {"keys": {}}
        key = add_key(keys_data, name="mykey")
        info = keys_data["keys"][key]
        assert info["name"] == "mykey"
        assert info["active"] is True
        assert "created" in info
        assert info["usage_count"] == 0

    def test_sets_quota_when_provided(self) -> None:
        """add_key should set quota when provided."""
        keys_data = {"keys": {}}
        key = add_key(keys_data, name="test", quota=1000)
        assert keys_data["keys"][key]["quota"] == 1000

    def test_creates_keys_dict_if_missing(self) -> None:
        """add_key should create 'keys' dict if missing."""
        keys_data = {}
        key = add_key(keys_data, name="test")
        assert "keys" in keys_data
        assert key in keys_data["keys"]


class TestRevokeKey:
    """Tests for revoke_key function."""

    def test_revokes_existing_key(self) -> None:
        """revoke_key should set active=False on existing key."""
        keys_data = {"keys": {"sk_dev_test": {"active": True}}}
        result = revoke_key(keys_data, "sk_dev_test")
        assert result is True
        assert keys_data["keys"]["sk_dev_test"]["active"] is False

    def test_returns_false_for_unknown_key(self) -> None:
        """revoke_key should return False for unknown key."""
        keys_data = {"keys": {}}
        result = revoke_key(keys_data, "sk_dev_unknown")
        assert result is False

    def test_handles_missing_keys_field(self) -> None:
        """revoke_key should return False if 'keys' field missing."""
        keys_data = {}
        result = revoke_key(keys_data, "sk_dev_test")
        assert result is False


class TestCheckRateLimit:
    """Tests for check_rate_limit function."""

    def test_within_quota_returns_true(self) -> None:
        """Should return True when usage_count < quota."""
        from trellis2_modal.service.auth import check_rate_limit

        key_info = {"usage_count": 5, "quota": 100}
        assert check_rate_limit(key_info) is True

    def test_at_quota_returns_false(self) -> None:
        """Should return False when usage_count >= quota."""
        from trellis2_modal.service.auth import check_rate_limit

        key_info = {"usage_count": 100, "quota": 100}
        assert check_rate_limit(key_info) is False

    def test_over_quota_returns_false(self) -> None:
        """Should return False when usage_count > quota."""
        from trellis2_modal.service.auth import check_rate_limit

        key_info = {"usage_count": 150, "quota": 100}
        assert check_rate_limit(key_info) is False

    def test_no_quota_returns_true(self) -> None:
        """Should return True when no quota is set (unlimited)."""
        from trellis2_modal.service.auth import check_rate_limit

        key_info = {"usage_count": 999999}
        assert check_rate_limit(key_info) is True

    def test_zero_quota_returns_false(self) -> None:
        """Should return False when quota is 0 (disabled key)."""
        from trellis2_modal.service.auth import check_rate_limit

        key_info = {"usage_count": 0, "quota": 0}
        assert check_rate_limit(key_info) is False
