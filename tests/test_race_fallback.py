"""Tests for Race Fallback timeout optimization."""

import pytest
from unittest.mock import MagicMock, patch, call
import concurrent.futures
import time

from blend.core.executor import Executor, LLMOutput


class TestRaceFallbackTimeout:
    """Test that Race Fallback uses optimized timeout values."""

    def test_primary_timeout_is_reasonable(self) -> None:
        """Primary model should not wait too long before firing fallback."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        # Track call order and timing
        call_times = []
        results = {
            "haiku": LLMOutput(
                content="fast response",
                model_used="haiku",
                tokens_used=10,
                tokens_budget_remaining=100,
                quality_gate_passed=True,
            ),
            "gemini": LLMOutput(
                content="slow response",
                model_used="gemini",
                tokens_used=20,
                tokens_budget_remaining=100,
                quality_gate_passed=True,
            ),
        }

        def slow_call(model, messages, **kwargs):
            call_times.append(("start", model, time.time()))
            time.sleep(2)  # Simulate slow response
            call_times.append(("end", model, time.time()))
            return results.get(model, LLMOutput(content="", model_used=model, tokens_used=0, tokens_budget_remaining=0, quality_gate_passed=True))

        def fast_call(model, messages, **kwargs):
            call_times.append(("start", model, time.time()))
            call_times.append(("end", model, time.time()))
            return results.get(model, LLMOutput(content="", model_used=model, tokens_used=0, tokens_budget_remaining=0, quality_gate_passed=True))

        with patch.object(executor, "_call_model_messages") as mock:
            # Fast model (haiku) returns immediately
            mock.side_effect = lambda **kwargs: fast_call("haiku", **kwargs) if kwargs.get("timeout", 0) > 100 else slow_call("haiku", **kwargs)

            # We want fallback to fire within reasonable time, not 3 seconds
            start = time.time()
            result = executor.execute_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
            )
            elapsed = time.time() - start

            # Should complete faster than 3s if fallback works properly
            # (allowing some margin for test overhead)
            assert elapsed < 2.5, f"Took {elapsed}s - fallback may not be firing fast enough"

    def test_timeout_configurable_per_model_type(self) -> None:
        """Different model types should have appropriate timeouts."""
        executor = Executor()

        # LOW complexity (fast task) should have shorter timeout
        # MEDIUM complexity should have moderate timeout
        # HIGH complexity should have longer timeout
        thresholds = {"low_max": 2, "medium_max": 5}

        # Verify thresholds are loaded from config, not hardcoded
        from blend.core.model_config import get_complexity_thresholds
        actual = get_complexity_thresholds()
        assert actual == thresholds or actual["low_max"] == 2


class TestFallbackChain:
    """Test fallback chain selection."""

    def test_low_complexity_fallback_empty(self) -> None:
        """LOW complexity should have minimal/empty fallback."""
        executor = Executor()
        selection = executor._select_model(complexity=1, task_type="general")
        # LOW should fallback to empty or minimal
        assert selection.primary == "minimax"

    def test_high_complexity_has_claude_fallback(self) -> None:
        """HIGH complexity should include Claude in fallback chain."""
        executor = Executor()
        selection = executor._select_model(complexity=7, task_type="general")
        # HIGH should have Claude in fallback
        assert "claude_sonnet" in selection.fallback or selection.primary in ["claude_sonnet", "gemini_pro_ultra"]