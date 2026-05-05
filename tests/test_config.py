"""Tests for config module."""

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_config_cache() -> Generator[None, None, None]:
    """Clear config cache before each test."""
    from blend.config import get_config

    get_config.cache_clear()
    yield


class TestConfigLoading:
    """Test configuration loading."""

    def test_minimax_api_key_from_env(self) -> None:
        """Minimax API key should come from MINIMAX_API_KEY env."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
            from blend.config import get_config

            config = get_config()
            assert config.MINIMAX_API_KEY == "test-key-123"

    def test_baosi_api_key_from_env(self) -> None:
        """Baosi API key should come from BAOSI_API_KEY env."""
        with patch.dict(os.environ, {"BAOSI_API_KEY": "sk-baosi-123"}):
            from blend.config import get_config

            config = get_config()
            assert config.BAOSI_API_KEY == "sk-baosi-123"

    def test_lemon_api_key_from_env(self) -> None:
        """Lemon API key should come from LEMON_API_KEY env."""
        with patch.dict(os.environ, {"LEMON_API_KEY": "lemon-key-456"}):
            from blend.config import get_config

            config = get_config()
            assert config.LEMON_API_KEY == "lemon-key-456"

    def test_port_default(self) -> None:
        """Port should default to 8000."""
        with patch.dict(
            os.environ,
            {"MINIMAX_API_KEY": "x", "ANTHROPIC_API_KEY": "x", "LEMON_API_KEY": "x"},
            clear=True,
        ):
            from blend.config import get_config

            config = get_config()
            assert config.PORT == 8000

    def test_port_from_env(self) -> None:
        """Port should be configurable via PORT env."""
        with patch.dict(
            os.environ,
            {
                "MINIMAX_API_KEY": "x",
                "ANTHROPIC_API_KEY": "x",
                "LEMON_API_KEY": "x",
                "PORT": "9000",
            },
        ):
            from blend.config import get_config

            config = get_config()
            assert config.PORT == 9000


class TestProviderInitialization:
    """Test provider initialization with env vars."""

    def test_minimax_provider_works_without_env_key(self) -> None:
        """MinimaxProvider initializes without API key (reads from env)."""
        with patch.dict(os.environ, {}, clear=True):
            from blend.providers.minimax_new import MinimaxProvider

            provider = MinimaxProvider()
            # No hardcoded fallback - key comes from env or empty
            assert provider._api_key == ""

    def test_minimax_provider_with_env_key(self) -> None:
        """MinimaxProvider should work with env var API key."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            from blend.providers.minimax_new import MinimaxProvider

            provider = MinimaxProvider()
            assert provider._api_key == "test-key"

    def test_baosi_provider_works_without_env_key(self) -> None:
        """BaosiProvider should work without API key (uses fallback)."""
        with patch.dict(os.environ, {}, clear=True):
            from blend.providers.baosiapi import BaosiProvider

            provider = BaosiProvider()
            # Should not raise, uses fallback empty key or env
            assert provider._api_key == ""

    def test_lemon_provider_works_without_env_key(self) -> None:
        """LemonProvider should work without API key (uses fallback)."""
        with patch.dict(os.environ, {}, clear=True):
            from blend.providers.lemonapi import LemonProvider

            provider = LemonProvider()
            # Should not raise, uses fallback empty key
            assert provider._api_key == ""


class TestConfigUtilities:
    """Test config utility functions."""

    def test_require_keys_raises_on_missing(self) -> None:
        """require_keys should raise ValueError when keys are missing."""
        with patch.dict(os.environ, {}, clear=True):
            from blend.config import require_keys

            with pytest.raises(ValueError) as exc_info:
                require_keys("MINIMAX_API_KEY", "BAOSI_API_KEY")
            assert "MINIMAX_API_KEY" in str(exc_info.value)
            assert "BAOSI_API_KEY" in str(exc_info.value)

    def test_require_keys_passes_when_all_present(self) -> None:
        """require_keys should pass when all keys are set."""
        with patch.dict(
            os.environ,
            {"MINIMAX_API_KEY": "key1", "BAOSI_API_KEY": "key2"},
        ):
            from blend.config import require_keys

            require_keys("MINIMAX_API_KEY", "BAOSI_API_KEY")  # Should not raise

    def test_validate_config_returns_errors_for_missing(self) -> None:
        """validate_config should return list of errors for missing keys."""
        with patch.dict(os.environ, {}, clear=True):
            from blend.config import get_config, validate_config

            get_config.cache_clear()
            errors = validate_config()
            assert isinstance(errors, list)
            assert len(errors) > 0
            assert any("MINIMAX_API_KEY" in e for e in errors)
            assert any("BAOSI_API_KEY" in e for e in errors)

    def test_validate_config_returns_empty_when_all_set(self) -> None:
        """validate_config should return empty list when all keys present."""
        with patch.dict(
            os.environ,
            {
                "MINIMAX_API_KEY": "key1",
                "BAOSI_API_KEY": "key2",
                "LEMON_API_KEY": "key3",
            },
        ):
            from blend.config import get_config, validate_config

            get_config.cache_clear()
            errors = validate_config()
            assert errors == []

    def test_get_config_dict_returns_all_fields(self) -> None:
        """get_config_dict should return dictionary with all config fields."""
        with patch.dict(
            os.environ,
            {
                "MINIMAX_API_KEY": "key1",
                "BAOSI_API_KEY": "key2",
                "LEMON_API_KEY": "key3",
                "PORT": "9000",
                "LOG_LEVEL": "DEBUG",
            },
        ):
            from blend.config import get_config_dict

            config_dict = get_config_dict()
            assert isinstance(config_dict, dict)
            assert "MINIMAX_API_KEY" in config_dict
            assert "BAOSI_API_KEY" in config_dict
            assert "LEMON_API_KEY" in config_dict
            assert "PORT" in config_dict
            assert "LOG_LEVEL" in config_dict
            assert config_dict["PORT"] == "9000"
            assert config_dict["LOG_LEVEL"] == "DEBUG"

    def test_port_default_with_empty_env(self) -> None:
        """Port should default to 8000 even with cleared env."""
        with patch.dict(
            os.environ,
            {
                "MINIMAX_API_KEY": "x",
                "ANTHROPIC_API_KEY": "x",
                "LEMON_API_KEY": "x",
            },
            clear=True,
        ):
            from blend.config import get_config

            config = get_config()
            assert config.PORT == 8000


class TestMCPServerConfig:
    """Test get_mcp_servers configuration."""

    def test_get_mcp_servers_empty_by_default(self) -> None:
        """get_mcp_servers returns empty list when BLEND_MCP_SERVERS is not set."""
        with patch.dict(os.environ, {}, clear=True):
            from blend.config import get_mcp_servers

            get_mcp_servers.cache_clear()
            assert get_mcp_servers() == []

    def test_get_mcp_servers_parses_json(self) -> None:
        """get_mcp_servers parses BLEND_MCP_SERVERS as JSON list."""
        import json

        config_json = json.dumps([
            {"name": "filesystem", "command": "npx", "args": ["-y", "@server/filesystem"]},
        ])
        with patch.dict(os.environ, {"BLEND_MCP_SERVERS": config_json}):
            from blend.config import get_mcp_servers

            get_mcp_servers.cache_clear()
            servers = get_mcp_servers()
            assert len(servers) == 1
            assert servers[0]["name"] == "filesystem"
            assert servers[0]["command"] == "npx"

    def test_get_mcp_servers_invalid_json_returns_empty(self) -> None:
        """get_mcp_servers returns empty list for invalid JSON."""
        with patch.dict(os.environ, {"BLEND_MCP_SERVERS": "not valid json {"}):
            from blend.config import get_mcp_servers

            get_mcp_servers.cache_clear()
            assert get_mcp_servers() == []

    def test_get_mcp_servers_non_list_returns_empty(self) -> None:
        """get_mcp_servers returns empty list if JSON is not a list."""
        with patch.dict(os.environ, {"BLEND_MCP_SERVERS": '"just a string"'}):
            from blend.config import get_mcp_servers

            get_mcp_servers.cache_clear()
            assert get_mcp_servers() == []
