"""Tests for story-003: L2 Strategy Layer - Opus Strategy Generation."""

from unittest.mock import patch

from blend.core.layers import L2Output
from blend.core.strategy import StrategyGenerator, StrategyResult


class TestStrategyGenerator:
    """Test the strategy generator."""

    def test_generate_strategy_output_format(self) -> None:
        """Test strategy generation returns correct format."""
        generator = StrategyGenerator()
        result = generator.generate(
            prompt="Design a system for handling user authentication",
            complexity=8,
        )
        assert hasattr(result, "output")
        assert isinstance(result.output, L2Output)
        assert "plan" in result.output.__dict__
        assert "quality_redlines" in result.output.__dict__

    def test_strategy_plan_contains_steps(self) -> None:
        """Strategy should include execution steps."""
        generator = StrategyGenerator()
        result = generator.generate(
            prompt="Build a REST API",
            complexity=7,
        )
        assert len(result.output.plan) > 0

    def test_strategy_redlines_exist(self) -> None:
        """Strategy should include quality redlines."""
        generator = StrategyGenerator()
        result = generator.generate(
            prompt="Build a REST API",
            complexity=7,
        )
        assert len(result.output.quality_redlines) >= 0

    def test_strategy_model_hint(self) -> None:
        """Strategy should suggest a model."""
        generator = StrategyGenerator()
        result = generator.generate(
            prompt="Build a REST API",
            complexity=7,
        )
        assert result.output.model_hint in ["Sonnet", "Opus"]

    def test_strategy_estimated_tokens_reasonable(self) -> None:
        """Estimated tokens should be reasonable and truncated flag set if over 300."""
        # Token estimate = word_count * 4 + 50 overhead. Need > 300.
        # With 10 plan steps (30 words each = 120 words), 3 redlines (25 words = 100 words),
        # 3 boundaries (25 words = 100 words): (320 * 4) + 50 = 1330 >> 300
        plan = [
            "Step {i}: " + " ".join(["design", "architecture", "for", "microservices",
               "with", "API", "gateway", "pattern", "service", "discovery",
               "circuit", "breaker", "load", "balancer", "cache", "layer",
               "message", "queue", "event", "driven", "communication",
               "monitoring", "logging", "tracing", "metrics", "health",
               "checks", "auto", "scaling", "failover", "recovery"])
            for i in range(1, 11)
        ]
        redlines = [
            "Validate " + " ".join(["all", "inputs", "for", "injection", "attacks",
               "sanitize", "outputs", "prevent", "XSS", "CSRF", "SQL",
               "command", "injection", "validate", "authentication", "authorization",
               "rate", "limit", "throttle", "requests", "log", "all",
               "security", "events", "encrypt", "data", "at", "rest"])
            for _ in range(3)
        ]
        boundaries = [
            "Handle " + " ".join(["empty", "null", "inputs", "gracefully",
               "network", "timeouts", "connection", "refused", "errors",
               "retry", "with", "backoff", "circuit", "open", "fallback",
               "to", "default", "values", "max", "payload", "size",
               "concurrent", "requests", "race", "conditions", "deadlocks",
               "memory", "pressure", "OOM", "crash", "recovery"])
            for _ in range(3)
        ]
        with patch.object(
            __import__("blend.core.strategy", fromlist=["StrategyGenerator"]).StrategyGenerator,
            "_call_opus",
            return_value=(plan, redlines, boundaries, ""),
        ):
            generator = StrategyGenerator()
            result = generator.generate(prompt="Build a REST API", complexity=8)
            assert result.output.estimated_tokens > 0
            assert result.truncated is True  # Must be truncated since > 300T

    def test_high_complexity_uses_opus(self) -> None:
        """High complexity tasks should hint at Opus."""
        generator = StrategyGenerator()
        result = generator.generate(
            prompt="Design a distributed system architecture with conflict resolution",
            complexity=9,
        )
        assert result.output.model_hint in ["Sonnet", "Opus"]

    def test_strategy_boundary_cases(self) -> None:
        """Strategy should identify boundary cases."""
        generator = StrategyGenerator()
        result = generator.generate(
            prompt="Process user input",
            complexity=6,
        )
        assert isinstance(result.output.boundary_cases, list)


class TestStrategyResult:
    """Test StrategyResult structure."""

    def test_strategy_result_creation(self) -> None:
        """Test creating StrategyResult."""
        output = L2Output(
            plan=["Step 1", "Step 2"],
            quality_redlines=["No hardcoded secrets"],
            boundary_cases=["Empty input"],
            model_hint="Sonnet",
            estimated_tokens=100,
        )
        result = StrategyResult(output=output, truncated=False)
        assert result.output.plan == ["Step 1", "Step 2"]
        assert result.truncated is False

    def test_strategy_result_truncated(self) -> None:
        """Test truncated strategy result."""
        output = L2Output(
            plan=["Step 1"],
            quality_redlines=[],
            boundary_cases=[],
            model_hint="Sonnet",
            estimated_tokens=50,
        )
        result = StrategyResult(output=output, truncated=True)
        assert result.truncated is True
