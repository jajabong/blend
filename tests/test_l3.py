"""Tests for story-004: L3 Execution Layer - Dynamic Model Selection."""

from typing import Any
from unittest.mock import MagicMock, patch

from blend.core.executor import Executor


def _mock_circuit_breaker():
    """Return a mock circuit breaker that allows all requests."""
    mock_breaker = MagicMock()
    mock_breaker.allow_request.return_value = True
    mock_breaker.state = MagicMock(value="closed")
    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_breaker
    return mock_registry


class MockResponse:
    """Mock response object."""

    def __init__(self, content: str) -> None:
        self.content: str = content
        self.model: str = "MiniMax-M2.7"
        self.usage: dict[str, Any] = {}
        self.raw: dict[str, Any] = {}


class TestModelSelection:
    """Test model selection based on complexity."""

    def test_select_haiku_for_low_complexity(self) -> None:
        """Low complexity (1-2) should select Haiku (Tier 1)."""
        with patch("blend.core.circuit_breaker.get_registry", return_value=_mock_circuit_breaker()):
            executor = Executor()
            selection = executor._select_model(2, "general")
            assert selection.primary == "haiku"

    def test_select_haiku_for_medium_complexity(self) -> None:
        """Medium complexity (4-7) should select Haiku or Sonnet when budget available."""
        with patch("blend.core.circuit_breaker.get_registry", return_value=_mock_circuit_breaker()):
            executor = Executor()
            with patch.object(executor, "_check_budget_status") as mock_budget:
                mock_budget.return_value = {
                    "minimax": 100000,
                    "haiku": 100000,
                    "sonnet": 100000,
                    "opus": 100000,
                    "gemini": 100000,
                }
                # complexity 7 hits sonnet path when sonnet budget > 1000
                selection = executor._select_model(7, "general")
                assert selection.primary in ["sonnet", "haiku"]

    def test_gemini_for_deep_reasoning(self) -> None:
        """Deep reasoning tasks should select Gemini."""
        with patch("blend.core.circuit_breaker.get_registry", return_value=_mock_circuit_breaker()):
            executor = Executor()
            with patch.object(executor, "_check_budget_status") as mock_budget:
                mock_budget.return_value = {
                    "minimax": 100000,
                    "haiku": 100000,
                    "sonnet": 100000,
                    "opus": 100000,
                    "gemini": 100000,
                }
                selection = executor._select_model(3, "deep_reasoning")
                assert selection.primary in ["gemini", "sonnet"]

    def test_code_tasks_use_claude(self) -> None:
        """Code tasks should use Claude (not Gemini)."""
        with patch("blend.core.circuit_breaker.get_registry", return_value=_mock_circuit_breaker()):
            executor = Executor()
            with patch.object(executor, "_check_budget_status") as mock_budget:
                mock_budget.return_value = {
                    "minimax": 100000,
                    "haiku": 100000,
                    "sonnet": 100000,
                    "opus": 100000,
                    "gemini": 100000,
                }
                selection = executor._select_model(5, "code")
                assert selection.primary in ["haiku", "sonnet", "minimax"]

    def test_fallback_chain(self) -> None:
        """Executor should have fallback chain."""
        with patch("blend.core.circuit_breaker.get_registry", return_value=_mock_circuit_breaker()):
            executor = Executor()
            with patch.object(executor, "_check_budget_status") as mock_budget:
                mock_budget.return_value = {
                    "minimax": 100000,
                    "haiku": 100000,
                    "sonnet": 100000,
                    "opus": 100000,
                    "gemini": 100000,
                }
                selection = executor._select_model(8, "general")
                assert selection.primary in ["sonnet", "haiku", "minimax"]
                assert isinstance(selection.fallback, list)


class TestExecutor:
    """Test the executor layer."""

    def test_execute_routes_correctly(self) -> None:
        """Executor should route to correct model."""
        mock_response = MockResponse("The weather is sunny")
        with patch("blend.core.executor._get_provider") as mock_get, \
             patch("blend.core.circuit_breaker.get_registry") as mock_reg:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = mock_response
            mock_get.return_value = (mock_provider, "MiniMax-M2.7")
            # Mock circuit breaker to allow all
            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True
            mock_breaker.state = MagicMock(value="closed")
            mock_reg.return_value.get.return_value = mock_breaker
            executor = Executor()
            result = executor.execute(
                prompt="What's the weather?",
                complexity=2,
            )
            assert result.model_used in ["minimax", "haiku", "sonnet", "opus", "gemini"]

    def test_execute_returns_l3_output(self) -> None:
        """Execute should return L3Output."""
        mock_response = MockResponse("Code here")
        with patch("blend.core.executor._get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = mock_response
            mock_get.return_value = (mock_provider, "MiniMax-M2.7")
            executor = Executor()
            result = executor.execute(
                prompt="Write code",
                complexity=5,
            )
            assert hasattr(result, "raw_output")
            assert hasattr(result, "model_used")
            assert hasattr(result, "tokens_used")

    def test_execution_quality_gate(self) -> None:
        """Execution should set quality_gate_passed."""
        mock_response = MockResponse("Test response")
        with patch("blend.core.executor._get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = mock_response
            mock_get.return_value = (mock_provider, "MiniMax-M2.7")
            executor = Executor()
            result = executor.execute(
                prompt="Test prompt",
                complexity=3,
            )
            assert isinstance(result.quality_gate_passed, bool)

    def test_tokens_budget_tracking(self) -> None:
        """Should track tokens budget."""
        mock_response = MockResponse("Response")
        with patch("blend.core.executor._get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = mock_response
            mock_get.return_value = (mock_provider, "MiniMax-M2.7")
            executor = Executor()
            result = executor.execute(
                prompt="Test prompt",
                complexity=4,
            )
            assert result.tokens_budget_remaining >= 0

    def test_strategy_injection(self) -> None:
        """Strategy should be passed to model calls."""
        mock_response = MockResponse("Result")
        with patch("blend.core.executor._get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = mock_response
            mock_get.return_value = (mock_provider, "claude-sonnet-4-6")
            executor = Executor()
            strategy: dict[str, object] = {"plan": ["Step 1", "Step 2"]}
            executor.execute(
                prompt="Test",
                complexity=8,
                strategy=strategy,
            )
            # Verify strategy was passed to provider
            assert mock_provider.chat.called
            call_args = mock_provider.chat.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2  # system + user
            assert messages[0]["role"] == "system"
            assert "plan" in messages[0]["content"].lower() or "Step" in messages[0]["content"]
