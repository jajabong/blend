"""Configuration module for blend - loads from environment variables."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment."""

    MINIMAX_API_KEYS: list[str]
    BAOSI_API_KEY: str
    LEMON_API_KEY: str
    PORT: int
    LOG_LEVEL: str

    @property
    def MINIMAX_API_KEY(self) -> str:
        """Backward compatibility: return the first key."""
        return self.MINIMAX_API_KEYS[0] if self.MINIMAX_API_KEYS else ""


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get application configuration.

    Loads from environment variables with defaults.

    Returns:
        Config instance

    Raises:
        ValueError: If required API keys are missing

    """
    # Support multiple Minimax keys
    minimax_keys_raw = os.environ.get("MINIMAX_API_KEYS", "")
    if minimax_keys_raw:
        minimax_keys = [k.strip() for k in minimax_keys_raw.split(",") if k.strip()]
    else:
        minimax_key = os.environ.get("MINIMAX_API_KEY", "")
        minimax_keys = [minimax_key] if minimax_key else []

    baosi_key = os.environ.get("BAOSI_API_KEY", "")
    lemon_key = os.environ.get("LEMON_API_KEY", "")
    port = int(os.environ.get("PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "INFO")

    return Config(
        MINIMAX_API_KEYS=minimax_keys,
        BAOSI_API_KEY=baosi_key,
        LEMON_API_KEY=lemon_key,
        PORT=port,
        LOG_LEVEL=log_level,
    )


def get_config_dict() -> dict[str, Any]:
    """Load all configuration as a dictionary.

    Returns:
        Dictionary with all configuration values

    """
    config = get_config()
    return {
        "MINIMAX_API_KEYS": config.MINIMAX_API_KEYS,
        "MINIMAX_API_KEY": config.MINIMAX_API_KEY,  # Backward compatibility
        "BAOSI_API_KEY": config.BAOSI_API_KEY,
        "LEMON_API_KEY": config.LEMON_API_KEY,
        "PORT": str(config.PORT),
        "LOG_LEVEL": config.LOG_LEVEL,
    }


def require_keys(*keys: str) -> None:
    """Check required configuration keys are present.

    Args:
        keys: Configuration key names to check

    Raises:
        ValueError: If any required key is missing or empty

    """
    config_dict = get_config_dict()
    missing = []
    for key in keys:
        val = config_dict.get(key)
        if not val:
            missing.append(key)
        elif isinstance(val, list) and not val:
            missing.append(key)

    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")


def validate_config() -> list[str]:
    """Validate configuration and return list of errors.

    Returns:
        List of error messages (empty if all valid)

    """
    errors: list[str] = []
    config = get_config()

    if not config.MINIMAX_API_KEYS:
        errors.append("MINIMAX_API_KEY or MINIMAX_API_KEYS is required")

    if not config.BAOSI_API_KEY:
        errors.append("BAOSI_API_KEY is required")

    if not config.LEMON_API_KEY:
        errors.append("LEMON_API_KEY is required")

    return errors


@lru_cache(maxsize=1)
def get_mcp_servers() -> list[dict[str, Any]]:
    """Load MCP server configurations from BLEND_MCP_SERVERS env var.

    Expected format: JSON list of {"name": "...", "command": "...", "args": [...]}

    Returns:
        List of MCP server configs, empty list if not configured

    """
    raw = os.environ.get("BLEND_MCP_SERVERS", "")
    if not raw:
        return []
    try:
        servers = json.loads(raw)
        if isinstance(servers, list):
            return servers
        return []
    except json.JSONDecodeError:
        return []
