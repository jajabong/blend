"""Tests for model_config module - YAML loading and configuration."""

from unittest.mock import patch

import pytest

from blend.core.model_config import (
    ModelConfig,
    ModelRegistry,
    get_fallback_chain,
    get_gemini_task_types,
    get_model_cost,
    get_model_map,
    load_model_registry,
)


class TestModelRegistryLoading:
    """Test model registry YAML loading."""

    def test_load_model_registry_returns_valid_registry(self) -> None:
        """load_model_registry should return a valid ModelRegistry."""
        registry = load_model_registry()
        assert isinstance(registry, ModelRegistry)
        assert isinstance(registry.tiers, dict)
        assert len(registry.tiers) > 0

    def test_tiers_have_required_fields(self) -> None:
        """Each tier should have required configuration fields."""
        registry = load_model_registry()
        for name, tier in registry.tiers.items():
            assert isinstance(tier, ModelConfig)
            assert tier.provider
            assert tier.model_name
            assert tier.tier
            assert tier.cost_per_1k >= 0

    def test_gemini_task_types_is_list(self) -> None:
        """gemini_task_types should be a list."""
        registry = load_model_registry()
        assert isinstance(registry.gemini_task_types, list)

    def test_degradation_rules_are_valid(self) -> None:
        """Degradation rules should have from/to model fields."""
        registry = load_model_registry()
        for rule in registry.degradation_rules:
            assert rule.from_model
            assert rule.to_model


class TestModelRegistryHelpers:
    """Test helper functions return correct types."""

    def test_get_model_map_returns_dict(self) -> None:
        """get_model_map should return dict of (provider, model_name) tuples."""
        model_map = get_model_map()
        assert isinstance(model_map, dict)
        assert len(model_map) > 0
        for key, value in model_map.items():
            assert isinstance(key, str)
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert isinstance(value[0], str)
            assert isinstance(value[1], str)

    def test_get_fallback_chain_returns_dict(self) -> None:
        """get_fallback_chain should return dict of fallback lists."""
        chain = get_fallback_chain()
        assert isinstance(chain, dict)
        for key, value in chain.items():
            assert isinstance(key, str)
            assert isinstance(value, list)

    def test_get_model_cost_returns_dict(self) -> None:
        """get_model_cost should return dict of costs."""
        costs = get_model_cost()
        assert isinstance(costs, dict)
        for key, value in costs.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))
            assert value >= 0

    def test_get_gemini_task_types_returns_collection(self) -> None:
        """get_gemini_task_types should return an iterable."""
        types = get_gemini_task_types()
        assert isinstance(types, (set, list, tuple))


class TestModelRegistryCache:
    """Test that registry is cached (lru_cache)."""

    def test_load_model_registry_is_cached(self) -> None:
        """load_model_registry should return same instance on repeated calls."""
        result1 = load_model_registry()
        result2 = load_model_registry()
        assert result1 is result2  # Same object due to caching


class TestModelRegistryErrorHandling:
    """Test error handling for missing files and invalid YAML."""

    def test_load_raises_when_yaml_not_found(self) -> None:
        """load_model_registry should raise when models.yaml not found."""
        with patch("blend.core.model_config._find_models_yaml") as mock_find:
            mock_find.side_effect = FileNotFoundError(
                "models.yaml not found"
            )
            load_model_registry.cache_clear()
            with pytest.raises(FileNotFoundError):
                load_model_registry()
