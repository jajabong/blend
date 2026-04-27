"""Additional tests for Executor module to improve coverage."""

from unittest.mock import MagicMock, patch

from blend.core.executor import Executor, LLMOutput


class TestExecutorEstimateTokens:
    """Test _estimate_tokens method."""

    def test_short_text(self) -> None:
        """Short text returns 0."""
        executor = Executor()
        result = executor._estimate_tokens("Hi")
        assert result == 0  # 2 chars / 4 = 0

    def test_medium_text(self) -> None:
        """Medium text returns correct estimate."""
        executor = Executor()
        result = executor._estimate_tokens("Hello world this is a test")
        # "Hello world this is a test" = 24 chars // 4 = 6
        assert result == 6


class TestExecutorGetBudget:
    """Test _get_budget method."""

    def test_all_models_have_budget(self) -> None:
        """All known models return a budget."""
        executor = Executor()
        assert executor._get_budget("minimax") == 100000
        assert executor._get_budget("haiku") == 200000
        assert executor._get_budget("sonnet") == 200000
        assert executor._get_budget("opus") == 200000
        assert executor._get_budget("gemini") == 200000

    def test_unknown_model_default_budget(self) -> None:
        """Unknown model returns default budget."""
        executor = Executor()
        assert executor._get_budget("unknown_model") == 200000


class TestExecutorSelectModelEdgeCases:
    """Test _select_model edge cases."""

    def test_complexity_zero_selects_haiku(self) -> None:
        """Complexity 0 (≤2 Tier1) should select Haiku."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000
        selection = executor._select_model(complexity=0, task_type="general")
        assert selection.primary == "haiku"

    def test_complexity_negative_treated_as_low(self) -> None:
        """Negative complexity (≤2 Tier1) should select Haiku."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000
        selection = executor._select_model(complexity=-1, task_type="general")
        assert selection.primary == "haiku"

    def test_complexity_very_high_treated_as_high(self) -> None:
        """Very high complexity (10+) treated as high."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 200 if m in ("sonnet", "haiku") else 10000
        )
        selection = executor._select_model(complexity=10, task_type="general")
        assert selection.primary == "sonnet"

    def test_tool_call_task_type_routes_to_gemini(self) -> None:
        """tool_call task type routes to Gemini."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 10000 if m == "gemini" else 0
        )
        selection = executor._select_model(complexity=5, task_type="tool_call")
        assert selection.primary == "gemini"

    def test_multimodal_task_type_routes_to_gemini(self) -> None:
        """multimodal task type routes to Gemini."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 10000 if m == "gemini" else 0
        )
        selection = executor._select_model(complexity=3, task_type="multimodal")
        assert selection.primary == "gemini"


class TestExecutorExecuteEdgeCases:
    """Test execute method edge cases."""

    def test_execute_with_strategy(self) -> None:
        """execute passes strategy to _call_model."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_provider.chat.return_value = MagicMock(content="result")

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "minimax")):
            result = executor.execute(
                prompt="test",
                complexity=5,
                strategy={"plan": ["Step 1: test", "Step 2: done"]},
            )

        assert result.raw_output == "result"
        # Verify strategy was injected
        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs["messages"][0]["role"] == "system"


class TestLLMOutputDataclass:
    """Test LLMOutput dataclass."""

    def test_llm_output_basic(self) -> None:
        """Basic LLMOutput creation."""
        output = LLMOutput(
            content="test",
            model_used="sonnet",
            tokens_used=100,
            tokens_budget_remaining=900,
            quality_gate_passed=True,
        )
        assert output.content == "test"
        assert output.finish_reason == "stop"  # default
        assert output.tool_calls is None  # default

    def test_llm_output_with_tool_calls(self) -> None:
        """LLMOutput with tool calls."""
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        output = LLMOutput(
            content="test",
            model_used="sonnet",
            tokens_used=100,
            tokens_budget_remaining=900,
            quality_gate_passed=True,
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )
        assert output.finish_reason == "tool_calls"
        assert output.tool_calls == tool_calls
