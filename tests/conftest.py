"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest

# Cassette directory for VCR recordings
CASSETTES_DIR = Path(__file__).parent / "cassettes"


@pytest.fixture(autouse=True)
def reset_circuit_breaker_registry():
    """Reset circuit breaker registry after each test to prevent state pollution."""
    yield
    # Reset the singleton registry after each test
    import blend.core.circuit_breaker as cb
    cb._registry = None


@pytest.fixture
def cassettes_dir():
    """Return the cassettes directory path."""
    CASSETTES_DIR.mkdir(exist_ok=True)
    return CASSETTES_DIR
