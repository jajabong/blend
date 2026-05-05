"""Tests for provider lifecycle management - Issue: Provider coupling."""

import pytest
from unittest.mock import patch, MagicMock

from blend.providers import MinimaxProvider, LemonProvider, BaosiProvider
from blend.providers.pool import ProviderPool, get_provider_pool


class TestProviderPool:
    """Test the ProviderPool singleton for proper instance reuse."""

    def test_provider_is_reused_via_pool(self):
        """Providers should be reused via pool, not recreated each call."""
        pool = get_provider_pool()

        # Get same model twice
        p1, name1 = pool.get("minimax", "MinimaxProvider", "MiniMax-M2.7")
        p2, name2 = pool.get("minimax", "MinimaxProvider", "MiniMax-M2.7")

        # Should be same instance
        assert p1 is p2, "Provider instances should be reused via pool"
        assert name1 == name2

        # Cleanup
        pool.close_all()

    def test_different_models_get_different_providers(self):
        """Different model keys should get different provider instances."""
        pool = get_provider_pool()

        p1, _ = pool.get("minimax", "MinimaxProvider", "MiniMax-M2.7")
        p2, _ = pool.get("sonnet", "BaosiProvider", "claude-sonnet-4-6")

        assert p1 is not p2

        pool.close_all()

    def test_provider_client_is_reused(self):
        """HTTP client (connection pool) should be reused."""
        pool = get_provider_pool()

        p1, _ = pool.get("minimax", "MinimaxProvider", "MiniMax-M2.7")
        p2, _ = pool.get("minimax", "MinimaxProvider", "MiniMax-M2.7")

        # Same provider instance means same HTTP client
        assert p1 is p2
        if hasattr(p1, '_client') and p1._client:
            assert p1._client is p2._client

        pool.close_all()

    def test_pool_stats(self):
        """Pool should track statistics."""
        pool = get_provider_pool()

        pool.get("minimax", "MinimaxProvider", "MiniMax-M2.7")
        pool.get("sonnet", "BaosiProvider", "claude-sonnet-4-6")

        stats = pool.stats
        assert stats["provider_count"] == 2
        assert stats["total_refs"] == 2

        pool.close_all()

    def test_close_all_closes_providers(self):
        """close_all() should close all providers."""
        pool = get_provider_pool()

        p1, _ = pool.get("minimax", "MinimaxProvider", "MiniMax-M2.7")
        p2, _ = pool.get("sonnet", "BaosiProvider", "claude-sonnet-4-6")

        with patch.object(p1, 'close') as mock_close1, \
             patch.object(p2, 'close') as mock_close2:
            pool.close_all()

            mock_close1.assert_called_once()
            mock_close2.assert_called_once()


class TestExecutorCleanup:
    """Test that Executor properly cleans up providers."""

    def test_executor_has_cleanup_method(self):
        """Executor should have cleanup() method."""
        from blend.core.executor import Executor

        executor = Executor()
        assert hasattr(executor, 'cleanup')
        assert callable(executor.cleanup)

        # Cleanup should not raise
        executor.cleanup()


class TestExecutorProviderPooling:
    """Test that Executor uses the provider pool."""

    def test_executor_uses_provider_pool(self):
        """Executor._get_provider should use the pool."""
        from blend.core.executor import Executor
        from blend.core.executor import _get_provider  # Module-level function

        # Clear the pool first
        pool = get_provider_pool()
        pool.close_all()

        executor = Executor()
        p1, _ = _get_provider("minimax")
        p2, _ = _get_provider("minimax")

        assert p1 is p2, "Executor should reuse providers via pool"

        executor.cleanup()