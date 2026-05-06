"""Provider Pool - Singleton provider registry for connection reuse."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blend.providers.base import LLMProvider


@dataclass
class ProviderHandle:
    """Handle to a pooled provider."""

    provider: LLMProvider
    model_name: str
    provider_class: str
    ref_count: int = 0


class ProviderPool:
    """Thread-safe singleton pool for provider instances.

    Reuses provider instances and their HTTP connection pools across
    multiple requests, reducing overhead from repeated instantiation.
    """

    _instance: ProviderPool | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._providers: dict[str, ProviderHandle] = {}
        self._pool_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ProviderPool:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get(
        self,
        model_key: str,
        provider_class_name: str,
        model_name: str,
    ) -> tuple[LLMProvider, str]:
        """Get or create a provider instance.

        Returns (provider, model_name) tuple.
        """
        with self._pool_lock:
            if model_key in self._providers:
                handle = self._providers[model_key]
                handle.ref_count += 1
                return handle.provider, handle.model_name

            # Create new instance
            provider = self._create_provider(provider_class_name)
            self._providers[model_key] = ProviderHandle(
                provider=provider,
                model_name=model_name,
                provider_class=provider_class_name,
                ref_count=1,
            )
            return provider, model_name

    def _create_provider(self, provider_class_name: str) -> LLMProvider:
        """Create a new provider instance."""
        # Import from blend.providers namespace to match test mocks
        from blend.providers import BaosiProvider, LemonProvider, MinimaxProvider

        if provider_class_name == "MinimaxProvider":
            return MinimaxProvider()
        elif provider_class_name == "LemonProvider":
            return LemonProvider()
        else:
            return BaosiProvider()

    def release(self, model_key: str) -> None:
        """Release a provider reference (close if no longer needed)."""
        with self._pool_lock:
            if model_key not in self._providers:
                return

            handle = self._providers[model_key]
            handle.ref_count -= 1

            if handle.ref_count <= 0:
                handle.provider.close()
                del self._providers[model_key]

    def close_all(self) -> None:
        """Close all providers and clear the pool."""
        with self._pool_lock:
            for handle in self._providers.values():
                handle.provider.close()
            self._providers.clear()

    @property
    def stats(self) -> dict[str, int]:
        """Get pool statistics."""
        with self._pool_lock:
            return {
                "provider_count": len(self._providers),
                "total_refs": sum(h.ref_count for h in self._providers.values()),
            }


def get_provider_pool() -> ProviderPool:
    """Get the global provider pool instance."""
    return ProviderPool.get_instance()


def reset_provider_pool() -> None:
    """Reset the singleton instance. For testing only."""
    with ProviderPool._lock:
        if ProviderPool._instance is not None:
            ProviderPool._instance.close_all()
            ProviderPool._instance = None
