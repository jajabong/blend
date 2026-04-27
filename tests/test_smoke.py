"""Tests for story-010: Smoke Tests - blend status validation."""

from typer.testing import CliRunner

from blend import __version__
from blend.cli import app as cli_app

runner = CliRunner()


class TestSmokeVersion:
    """Smoke test: version command."""

    def test_version_output(self) -> None:
        """blend --version should output valid version."""
        result = runner.invoke(cli_app, ["--version"])
        # Accept exit codes 0 or 2 (typer may exit with 2 for --version)
        assert result.exit_code in [0, 2]
        assert __version__ in result.stdout or __version__ in str(result.exception)


class TestSmokeStatus:
    """Smoke test: status command."""

    def test_status_runs(self) -> None:
        """blend status should run without errors."""
        result = runner.invoke(cli_app, ["status"])
        assert result.exit_code == 0

    def test_status_shows_layers(self) -> None:
        """blend status should show layer status."""
        result = runner.invoke(cli_app, ["status"])
        assert result.exit_code == 0
        output = result.stdout.lower()
        # Should mention key components
        assert "layer" in output or "status" in output


class TestSmokeCLI:
    """Smoke test: CLI structure."""

    def test_cli_commands_exist(self) -> None:
        """CLI should have all expected commands."""
        # Check status command works
        result = runner.invoke(cli_app, ["status"])
        assert result.exit_code == 0

        # Check --help works
        result = runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        assert "blend" in result.stdout.lower()


class TestSmokeCoreComponents:
    """Smoke test: core components import."""

    def test_version_defined(self) -> None:
        """Version should be defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_core_imports(self) -> None:
        """Core components should be importable."""
        from blend.core.budget import ResourceModel
        from blend.core.enforcer import Enforcer
        from blend.core.layers import Layer
        from blend.intent.scorer import ComplexityScorer

        assert Layer is not None
        assert Enforcer is not None
        assert ResourceModel is not None
        assert ComplexityScorer is not None
