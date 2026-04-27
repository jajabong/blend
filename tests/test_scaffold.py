"""Tests for story-001: Project Scaffold."""

from blend import __version__


def test_version_is_string() -> None:
    """Version should be a valid string."""
    assert isinstance(__version__, str)


def test_version_format() -> None:
    """Version should follow semver format."""
    parts = __version__.split(".")
    assert len(parts) == 3, f"Version {__version__} should be X.Y.Z format"
    for part in parts:
        assert part.isdigit(), f"Version part {part} should be digit"


def test_version_value() -> None:
    """Version should be defined and non-empty."""
    assert __version__ is not None
    assert len(__version__) > 0
