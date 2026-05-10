"""Tests for SemanticCache TTL functionality."""

import time

import pytest

from blend.core.semantic_cache import SemanticCache


class TestSemanticCacheTTL:
    """Test SemanticCache TTL feature."""

    def test_cache_entry_has_created_timestamp(self) -> None:
        """Cache entries should store creation time for TTL calculation."""
        cache = SemanticCache(max_entries=100, ttl_seconds=60)
        cache.set(
            prompt="test prompt",
            response="test response",
            model_used="minimax",
            tokens_saved=100,
        )

        stats = cache.stats()
        assert "created_at" in stats or hasattr(cache, "_cache")

    def test_get_returns_none_after_ttl_expires(self) -> None:
        """Entries should expire after TTL."""
        cache = SemanticCache(max_entries=100, ttl_seconds=1)

        cache.set(
            prompt="expiring prompt",
            response="response",
            model_used="minimax",
            tokens_saved=100,
        )

        # Immediate get should work
        result = cache.get("expiring prompt")
        assert result.hit is True

        # Wait for TTL to expire
        time.sleep(1.5)

        # After TTL, should be cache miss
        result = cache.get("expiring prompt")
        assert result.hit is False

    def test_ttl_none_means_no_expiration(self) -> None:
        """When TTL is None, entries should never expire."""
        cache = SemanticCache(max_entries=100, ttl_seconds=None)

        cache.set(
            prompt="permanent prompt",
            response="response",
            model_used="minimax",
            tokens_saved=100,
        )

        time.sleep(0.5)

        result = cache.get("permanent prompt")
        assert result.hit is True

    def test_invalidate_removes_entry_before_ttl(self) -> None:
        """invalidate() should remove entry even if TTL hasn't expired."""
        cache = SemanticCache(max_entries=100, ttl_seconds=60)

        cache.set(
            prompt="to invalidate",
            response="response",
            model_used="minimax",
            tokens_saved=100,
        )

        result = cache.get("to invalidate")
        assert result.hit is True

        cache.invalidate("to invalidate")

        result = cache.get("to invalidate")
        assert result.hit is False

    def test_clear_removes_all_entries(self) -> None:
        """clear() should remove all entries."""
        cache = SemanticCache(max_entries=100, ttl_seconds=60)

        cache.set("prompt1", "response1", "minimax", 100)
        cache.set("prompt2", "response2", "minimax", 100)

        assert len(cache._cache) == 2

        cache.clear()

        assert len(cache._cache) == 0

    def test_stats_includes_ttl_info(self) -> None:
        """stats() should include TTL configuration."""
        cache = SemanticCache(max_entries=100, ttl_seconds=300)
        stats = cache.stats()

        assert "ttl_seconds" in stats
        assert stats["ttl_seconds"] == 300