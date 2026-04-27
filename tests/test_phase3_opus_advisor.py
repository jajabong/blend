"""Tests for Phase 3: Opus-driven model_hint in L2 Strategy Layer.

TDD Phase:
- RED: Write failing tests for Opus-driven model_recommendation
- GREEN: Implement changes to strategy.py
- IMPROVE: Verify all pass
"""

from unittest.mock import patch

from blend.core.strategy import OPUS_SYSTEM_PROMPT, StrategyGenerator


class TestOpusModelRecommendation:
    """Test that Opus generates model_recommendation in JSON output."""

    def test_opus_prompt_includes_model_recommendation_field(self) -> None:
        """OPUS_SYSTEM_PROMPT must request model_recommendation field."""
        assert "model_recommendation" in OPUS_SYSTEM_PROMPT, (
            "OPUS_SYSTEM_PROMPT must request model_recommendation in JSON output"
        )

    def test_opus_response_with_model_recommendation(self) -> None:
        """When Opus returns model_recommendation, it should be used as model_hint."""
        generator = StrategyGenerator()
        # Patch _call_opus directly to return Opus's model recommendation
        with patch.object(
            generator,
            "_call_opus",
            return_value=(["Step 1"], ["No SQL injection"], ["Empty input"], "Opus"),
        ):
            result = generator.generate(
                prompt="Design a distributed system",
                complexity=8,
            )
            assert result.output.model_hint == "Opus"

    def test_opus_response_sonnet_recommendation(self) -> None:
        """Opus can recommend Sonnet even for high complexity."""
        generator = StrategyGenerator()
        with patch.object(
            generator,
            "_call_opus",
            return_value=(["Step 1"], ["Check input"], ["Edge case"], "Sonnet"),
        ):
            result = generator.generate(
                prompt="Write a simple function",
                complexity=9,
            )
            assert result.output.model_hint == "Sonnet"

    def test_opus_response_haiku_recommendation(self) -> None:
        """Opus can recommend Haiku when appropriate."""
        generator = StrategyGenerator()
        with patch.object(
            generator,
            "_call_opus",
            return_value=(["Step 1"], [], [], "Haiku"),
        ):
            result = generator.generate(
                prompt="Simple task",
                complexity=6,
            )
            assert result.output.model_hint == "Haiku"


class TestOpusFallbackModelHint:
    """Test fallback when Opus doesn't return model_recommendation."""

    def test_no_model_recommendation_falls_back_to_rules(self) -> None:
        """When Opus JSON lacks model_recommendation, fall back to rule-based."""
        generator = StrategyGenerator()
        # Return empty string as model_recommendation (simulates missing field)
        with patch.object(
            generator,
            "_call_opus",
            return_value=(["Step 1"], ["No SQL injection"], ["Empty input"], ""),
        ):
            result = generator.generate(
                prompt="Design a distributed system",
                complexity=9,
            )
            # Empty recommendation → fallback rule: complexity >= 9 → "Opus"
            assert result.output.model_hint == "Opus"

    def test_opus_failure_falls_back_to_rules(self) -> None:
        """When Opus call fails, fall back to rule-based model_hint."""
        generator = StrategyGenerator()
        # Make _call_opus raise so fallback is triggered
        with patch.object(generator, "_call_opus", side_effect=Exception("API error")):
            result = generator.generate(
                prompt="Design a system",
                complexity=9,
            )
            # Fallback to rule-based: complexity >= 9 → "Opus"
            assert result.output.model_hint == "Opus"
            # Verify plan was still generated via rule-based fallback
            assert len(result.output.plan) > 0


class TestLowComplexityNoOpus:
    """Test that low complexity bypasses Opus and uses rule-based model_hint."""

    def test_complexity_5_uses_rule_based_no_opus(self) -> None:
        """Complexity < 6 does not call Opus, uses rule-based model_hint."""
        generator = StrategyGenerator()

        with patch("blend.core.strategy.StrategyGenerator._call_opus") as mock_opus:
            result = generator.generate(
                prompt="Simple task",
                complexity=5,
            )

            # _call_opus should NOT be called for complexity < 6
            mock_opus.assert_not_called()
            # Rule-based: complexity < 9 → "Sonnet"
            assert result.output.model_hint == "Sonnet"


class TestModelRecommendationValidValues:
    """Test that model_recommendation accepts valid model names."""

    def test_valid_model_names_accepted(self) -> None:
        """model_recommendation must accept Opus/Sonnet/Haiku."""
        valid_models = ["Opus", "Sonnet", "Haiku"]

        for model_name in valid_models:
            generator = StrategyGenerator()
            with patch.object(
                generator,
                "_call_opus",
                return_value=(["Step 1"], [], [], model_name),
            ):
                result = generator.generate(
                    prompt="Task",
                    complexity=8,
                )

                assert result.output.model_hint == model_name, (
                    f"Failed for model: {model_name}"
                )
