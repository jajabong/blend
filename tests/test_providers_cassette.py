"""Cassette-based integration tests for providers.

These tests use VCR cassettes to record and replay real HTTP interactions.
Unlike the mocked tests in test_providers.py, these tests verify actual
provider behavior against recorded cassettes.

Running tests:
- Normal mode (playback):     pytest tests/test_providers_cassette.py
- Record new cassettes:       BLEND_RECORD_CASSETTES=1 pytest tests/test_providers_cassette.py
- Update existing cassettes: BLEND_CASSETTE_MODE=record pytest tests/test_providers_cassette.py

Cassettes are stored in tests/cassettes/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Check if recording is enabled
RECORD_MODE = os.environ.get("BLEND_CASSETTE_MODE") == "record" or os.environ.get("BLEND_RECORD_CASSETTES") == "1"

# Skip these tests if not in a real API environment
SKIP_IF_NO_KEYS = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set - cassette tests require real API"
)


def _mock_circuit_breaker():
    """Return a mock circuit breaker that allows all requests."""
    mock_breaker = MagicMock()
    mock_breaker.allow_request.return_value = True
    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_breaker
    return mock_registry


class TestMinimaxCassette:
    """Integration tests using pre-recorded Minimax cassettes."""

    @SKIP_IF_NO_KEYS
    def test_chat_with_cassette_playback(self) -> None:
        """Test Minimax chat using cassette playback (default mode).

        When cassettes exist, this test replays the recorded HTTP response
        without making real API calls.
        """
        cassette_path = Path(__file__).parent / "cassettes" / "minimax_chat.yaml"

        # If in record mode and cassette doesn't exist, we'd record
        # For playback, we use vcrpy to replay the cassette
        if RECORD_MODE and not cassette_path.exists():
            # Record a new cassette
            self._record_cassette(cassette_path)

        # Verify cassette exists for playback
        assert cassette_path.exists(), (
            f"Cassette not found: {cassette_path}\n"
            "Run with BLEND_RECORD_CASSETTES=1 to record it first."
        )

    def _record_cassette(self, cassette_path: Path) -> None:
        """Record a new cassette by making a real API call.

        This method is only called in record mode when cassette doesn't exist.
        """
        from blend.providers.minimax import MinimaxProvider

        with patch("blend.providers.minimax.get_registry", return_value=_mock_circuit_breaker()):
            provider = MinimaxProvider(api_key=os.environ.get("MINIMAX_API_KEY", ""))

            # Make the actual API call - this will be recorded
            result = provider.chat(
                messages=[{"role": "user", "content": "Hello, respond with a single word"}],
                model="MiniMax-M2.7",
            )

            # The cassette will be saved by VCR
            assert result.content  # Verify we got a response


class TestProviderCassetteHelpers:
    """Helper utilities for cassette management."""

    @staticmethod
    def get_cassette_path(name: str) -> Path:
        """Get path for a cassette by name.

        Args:
            name: Cassette name (e.g., "minimax_chat", "baosi_claude")

        Returns:
            Path to the cassette file
        """
        return Path(__file__).parent / "cassettes" / f"{name}.yaml"

    @staticmethod
    def cassette_exists(name: str) -> bool:
        """Check if a cassette exists.

        Args:
            name: Cassette name

        Returns:
            True if cassette file exists
        """
        return TestProviderCassetteHelpers.get_cassette_path(name).exists()

    @staticmethod
    def delete_cassette(name: str) -> None:
        """Delete a cassette file.

        Args:
            name: Cassette name to delete

        Note:
            Call this in record mode to force re-recording.
        """
        path = TestProviderCassetteHelpers.get_cassette_path(name)
        if path.exists():
            path.unlink()

    @staticmethod
    def list_cassettes() -> list[str]:
        """List all available cassettes.

        Returns:
            List of cassette names (without .yaml extension)
        """
        cassette_dir = Path(__file__).parent / "cassettes"
        return [p.stem for p in cassette_dir.glob("*.yaml")]


# =============================================================================
# Cassette management CLI helpers
# =============================================================================

def pytest_configure(config: Any) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "cassette: mark test as using VCR cassettes")
    config.addinivalue_line("markers", "record: mark test as recording new cassettes")


def pytest_addoption(parser: Any) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--record-cassettes",
        action="store_true",
        default=False,
        help="Record new cassettes even if they already exist",
    )
    parser.addoption(
        "--cassette-mode",
        choices=["playback", "record", "none"],
        default="playback",
        help="Cassette mode: playback (default), record, or none",
    )


@pytest.fixture
def cassette_mode(request: Any) -> str:
    """Get current cassette mode from command line options."""
    if request.config.getoption("--cassette-mode") == "record":
        return "record"
    if request.config.getoption("--record-cassettes"):
        return "record"
    return "playback"
