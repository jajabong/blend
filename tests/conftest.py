"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture(autouse=True)
def reset_circuit_breaker_registry():
    """Reset circuit breaker registry after each test to prevent state pollution."""
    yield
    # Reset the singleton registry after each test
    import blend.core.circuit_breaker as cb
    cb._registry = None
