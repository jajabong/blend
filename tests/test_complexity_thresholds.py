"""Tests for complexity threshold configuration from models.yaml."""

import pytest

from blend.core.model_config import (
    get_complexity_thresholds,
    load_model_registry,
)


class TestComplexityThresholds:
    """Test complexity thresholds are loaded from models.yaml."""

    def test_get_complexity_thresholds_returns_dict(self) -> None:
        """get_complexity_thresholds returns a dict with threshold values."""
        thresholds = get_complexity_thresholds()
        assert isinstance(thresholds, dict)
        assert "low_max" in thresholds
        assert "medium_max" in thresholds

    def test_low_max_is_2(self) -> None:
        """LOW complexity threshold should be 2."""
        thresholds = get_complexity_thresholds()
        assert thresholds["low_max"] == 2

    def test_medium_max_is_5(self) -> None:
        """MEDIUM complexity threshold should be 5."""
        thresholds = get_complexity_thresholds()
        assert thresholds["medium_max"] == 5

    def test_thresholds_match_yaml(self) -> None:
        """Thresholds should match what is defined in models.yaml."""
        registry = load_model_registry()
        routing = registry.complexity_routing

        # Get thresholds from registry
        low_cfg = routing.get("low")
        medium_cfg = routing.get("medium")

        assert low_cfg is not None
        assert medium_cfg is not None

        # The YAML uses max_complexity
        # For low: max_complexity: 2 means complexity <= 2 is LOW
        # For medium: max_complexity: 5 means complexity <= 5 is MEDIUM

        thresholds = get_complexity_thresholds()
        assert thresholds["low_max"] == low_cfg.max_complexity
        assert thresholds["medium_max"] == medium_cfg.max_complexity