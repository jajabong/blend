"""Tests for blend CLI."""

from typer.testing import CliRunner

from blend import __version__
from blend.cli import app

runner = CliRunner()


def test_version_command() -> None:
    """Test version command outputs version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_status_command() -> None:
    """Test status command runs smoke test."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "OK" in result.stdout
    assert "All checks passed" in result.stdout
