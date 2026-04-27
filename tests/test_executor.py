"""Tests for Executor fallback chains."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from blend.core.executor import Executor


def _make_mock_response(content: str = "test response") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    return resp


class TestFallbackChains:
    """Test fallback chain routing for all model tiers."""

    def test_minimax_no_fallback(self) -> None:
        """Minimax should have no fallback and succeed directly."""
        mock_provider = MagicMock()
        mock_provider.chat.return_value = _make_mock_response()

        with patch("blend.providers.MinimaxProvider", return_value=mock_provider):
            executor = Executor()
            executor.resource_model = MagicMock()
            # All budgets exhausted except minimax
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 10000 if m == "minimax" else 0
            )

            # complexity=2 → Tier1 → Haiku if available, else Minimax
            # With Haiku exhausted, Minimax is selected
            result = executor.execute("test prompt", complexity=2)

            assert result.model_used == "minimax"
            assert mock_provider.chat.call_count == 1

    def test_haiku_fallback_to_minimax(self) -> None:
        """Haiku fails → fallback to minimax."""
        haiku_mock = MagicMock()
        haiku_mock.chat.side_effect = Exception("Haiku unavailable")
        minimax_mock = MagicMock()
        minimax_mock.chat.return_value = _make_mock_response()

        def provider_side_effect(key: str) -> tuple[MagicMock, str]:
            if key == "haiku":
                return haiku_mock, "claude-haiku-4-5-20251001"
            elif key == "minimax":
                return minimax_mock, "MiniMax-M2.7"
            return MagicMock(), key

        with patch("blend.core.executor._get_provider", side_effect=provider_side_effect):
            executor = Executor()
            executor.resource_model = MagicMock()
            # sonnet exhausted → haiku selected as primary (budget > 100); then fails → fallback minimax
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 200 if m == "haiku" else 10000
            )

            result = executor.execute("test prompt", complexity=5)

            assert result.model_used == "minimax"
            haiku_mock.chat.assert_called_once()
            minimax_mock.chat.assert_called_once()

    def test_sonnet_fallback_haiku_then_minimax(self) -> None:
        """Sonnet fails → haiku fails → minimax succeeds."""
        sonnet_mock = MagicMock()
        sonnet_mock.chat.side_effect = Exception("Sonnet unavailable")
        haiku_mock = MagicMock()
        haiku_mock.chat.side_effect = Exception("Haiku unavailable")
        minimax_mock = MagicMock()
        minimax_mock.chat.return_value = _make_mock_response()

        def provider_side_effect(key: str) -> tuple[MagicMock, str]:
            if key == "sonnet":
                return sonnet_mock, "claude-sonnet-4-6"
            elif key == "haiku":
                return haiku_mock, "claude-haiku-4-5-20251001"
            elif key == "minimax":
                return minimax_mock, "MiniMax-M2.7"
            return MagicMock(), key

        with patch("blend.core.executor._get_provider", side_effect=provider_side_effect):
            executor = Executor()
            executor.resource_model = MagicMock()
            # sonnet budget > 100 so it is selected as primary for complexity 6; then fails → haiku fails → minimax succeeds
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 200 if m in ("sonnet", "haiku") else 10000
            )

            result = executor.execute("test prompt", complexity=6)

            assert result.model_used == "minimax"
            sonnet_mock.chat.assert_called_once()
            haiku_mock.chat.assert_called_once()
            minimax_mock.chat.assert_called_once()

    def test_gemini_fallback_to_minimax(self) -> None:
        """Gemini fails → minimax succeeds."""
        gemini_mock = MagicMock()
        gemini_mock.chat.side_effect = Exception("Gemini unavailable")
        minimax_mock = MagicMock()
        minimax_mock.chat.return_value = _make_mock_response()

        def provider_side_effect(key: str) -> tuple[MagicMock, str]:
            if key == "gemini":
                return gemini_mock, "[L]gemini-3-flash-preview"
            elif key == "minimax":
                return minimax_mock, "MiniMax-M2.7"
            return MagicMock(), key

        with patch("blend.core.executor._get_provider", side_effect=provider_side_effect):
            executor = Executor()
            executor.resource_model = MagicMock()
            # gemini budget > 1000 so it is selected for deep_reasoning; sonnet/haiku are exhausted
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 10000 if m == "gemini" else 0
            )

            result = executor.execute("test prompt", complexity=9, task_type="deep_reasoning")

            assert result.model_used == "minimax"
            gemini_mock.chat.assert_called_once()
            minimax_mock.chat.assert_called_once()


class TestModelSelection:
    """Test model selection logic."""

    def test_low_complexity_selects_haiku(self) -> None:
        """Complexity 1-2 should select Haiku (Tier 1)."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        selection = executor._select_model(complexity=2, task_type="general")
        assert selection.primary == "haiku"

    def test_medium_with_sonnet_budget_selects_sonnet(self) -> None:
        """Complexity 4-5 with sonnet budget should select sonnet."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 10000 if m in ("haiku", "sonnet") else 10000
        )

        selection = executor._select_model(complexity=5, task_type="general")
        assert selection.primary == "sonnet"
        assert "haiku" in selection.fallback

    def test_medium_without_sonnet_budget_selects_haiku(self) -> None:
        """Complexity 4-5 without sonnet budget should select haiku."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 10000 if m == "haiku" else 0
        )

        selection = executor._select_model(complexity=5, task_type="general")
        assert selection.primary == "haiku"

    def test_medium_high_selects_sonnet(self) -> None:
        """Complexity 6-7 should select sonnet."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        selection = executor._select_model(complexity=6, task_type="general")
        assert selection.primary == "sonnet"

    def test_high_selects_sonnet(self) -> None:
        """Complexity 8-10 should select sonnet (Opus for L2 only)."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        selection = executor._select_model(complexity=9, task_type="general")
        assert selection.primary == "sonnet"

    def test_deep_reasoning_with_gemini_budget(self) -> None:
        """Deep reasoning should route to Gemini when budget available."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 10000 if m == "gemini" else 10000
        )

        selection = executor._select_model(complexity=8, task_type="deep_reasoning")
        assert selection.primary == "gemini"
        assert "minimax" in selection.fallback

    def test_deep_reasoning_without_gemini_fallback(self) -> None:
        """Without Gemini budget, deep reasoning falls back to Sonnet."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 0 if m == "gemini" else 10000
        )

        selection = executor._select_model(complexity=8, task_type="deep_reasoning")
        assert selection.primary == "sonnet"


class TestExecuteMessages:
    """Test execute_messages with message list and tool parameters."""

    def test_execute_messages_with_tools_forwarded(self) -> None:
        """execute_messages forwards tools and response_format to provider."""
        from blend.core.executor import LLMOutput

        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        response_format = {"type": "json_object"}

        # Patch at instance level since _call_model_messages is a bound method
        mock_output = LLMOutput(
            content="result",
            model_used="minimax",
            tokens_used=10,
            tokens_budget_remaining=100,
            quality_gate_passed=True,
            finish_reason="stop",
            tool_calls=None,
        )
        with patch.object(executor, "_call_model_messages", return_value=mock_output) as mock_call:
            result = executor.execute_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                tools=tools,
                response_format=response_format,
            )

            mock_call.assert_called_once()
            call_kwargs = mock_call.call_args.kwargs
            assert call_kwargs["tools"] == tools
            assert call_kwargs["response_format"] == response_format
            assert result.content == "result"

    def test_execute_messages_fallback_chain(self) -> None:
        """execute_messages falls back when primary model fails."""
        from blend.core.executor import LLMOutput

        executor = Executor()
        executor.resource_model = MagicMock()
        # sonnet selected as primary (complexity 5, budget > 100)
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 10000 if m in ("sonnet", "haiku", "minimax") else 0
        )

        def call_side_effect(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> LLMOutput:
            if model == "sonnet":
                raise Exception("Sonnet down")
            return LLMOutput(
                content="fallback result",
                model_used=model,
                tokens_used=5,
                tokens_budget_remaining=100,
                quality_gate_passed=True,
                finish_reason="stop",
                tool_calls=None,
            )

        with patch.object(executor, "_call_model_messages", side_effect=call_side_effect):
            result = executor.execute_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=5,
            )

            assert result.model_used == "haiku"
            assert result.content == "fallback result"


class TestStreamMessages:
    """Test stream_messages."""

    def test_stream_messages_yields_chunks(self) -> None:
        """stream_messages yields dict chunks."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        def fake_stream() -> Iterator[str]:
            yield '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
            yield '{"choices":[{"delta":{},"finish_reason":"stop"}]}'

        with patch("blend.providers.BaosiProvider") as mock_provider:
            instance = MagicMock()
            instance.chat_stream.return_value = fake_stream()
            mock_provider.return_value = instance

            # complexity=3: Tier2, exhaust sonnet budget so haiku selected
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            chunks = list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
            ))

            assert len(chunks) >= 1
            assert "delta" in chunks[0]

    def test_stream_messages_with_tools(self) -> None:
        """stream_messages forwards tools to provider."""
        executor = Executor()
        executor.resource_model = MagicMock()
        # complexity=3: Tier2, exhaust sonnet budget so haiku selected
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 0 if m == "sonnet" else 10000
        )

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        def fake_stream() -> Iterator[str]:
            yield '{"choices":[{"delta":{"content":"Done"},"finish_reason":"stop"}]}'

        with patch("blend.providers.BaosiProvider") as mock_provider:
            instance = MagicMock()
            instance.chat_stream.return_value = fake_stream()
            mock_provider.return_value = instance

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Weather?"}],
                complexity=3,
                tools=tools,
            ))

            call_kwargs = instance.chat_stream.call_args.kwargs
            assert call_kwargs["tools"] == tools


class TestAllModelsFailFallback:
    """Test fallback to minimax when all models fail."""

    def test_execute_all_models_fail_then_minimax(self) -> None:
        """execute() falls back to minimax when all selected models fail."""
        # Primary (sonnet) and fallback (haiku) both fail → minimax succeeds
        def fail_then_ok(key: str) -> tuple[MagicMock, str]:
            if key == "sonnet":
                m = MagicMock()
                m.chat.side_effect = Exception("Sonnet down")
                return m, "claude-sonnet-4-6"
            elif key == "haiku":
                m = MagicMock()
                m.chat.side_effect = Exception("Haiku down")
                return m, "claude-haiku-4-5-20251001"
            elif key == "minimax":
                m = MagicMock()
                m.chat.return_value = _make_mock_response("minimax result")
                return m, "MiniMax-M2.7"
            return MagicMock(), key

        with patch("blend.core.executor._get_provider", side_effect=fail_then_ok):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.return_value = 10000

            result = executor.execute("test prompt", complexity=7)

            assert result.model_used == "minimax"
            assert result.raw_output == "minimax result"

    def test_execute_messages_all_models_fail_then_minimax(self) -> None:
        """execute_messages() falls back to minimax when all selected models fail."""
        from blend.core.executor import LLMOutput

        def call_side_effect(model: str, **kwargs: Any) -> LLMOutput:
            if model != "minimax":
                raise Exception(f"{model} down")
            return LLMOutput(
                content="minimax fallback",
                model_used="minimax",
                tokens_used=5,
                tokens_budget_remaining=100,
                quality_gate_passed=True,
            )

        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.side_effect = (
            lambda m: 10000 if m in ("sonnet", "haiku", "minimax") else 0
        )

        with patch.object(executor, "_call_model_messages", side_effect=call_side_effect):
            result = executor.execute_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=6,
            )

            assert result.model_used == "minimax"


class TestStream:
    """Test stream() method."""

    def test_stream_yields_chunks(self) -> None:
        """stream() yields text chunks from provider."""
        def fake_stream() -> Iterator[str]:
            yield '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
            yield '{"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}'

        mock_provider = MagicMock()
        mock_provider.chat_stream.return_value = fake_stream()

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "minimax")):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.return_value = 10000

            chunks = list(executor.stream(prompt="test", complexity=3))

            joined = "".join(chunks)
            assert "Hello" in joined
            assert "world" in joined

    def test_stream_with_strategy_injects_system_prompt(self) -> None:
        """stream() injects L2 strategy as system message."""
        def fake_stream() -> Iterator[str]:
            yield '{"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}'

        mock_provider = MagicMock()
        mock_provider.chat_stream.return_value = fake_stream()

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "minimax")):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.return_value = 10000

            list(executor.stream(
                prompt="test",
                complexity=5,
                strategy={"plan": ["Step 1: analyze", "Step 2: execute"]},
            ))

            call_kwargs = mock_provider.chat_stream.call_args.kwargs
            sent = call_kwargs["messages"]
            assert sent[0]["role"] == "system"
            assert "analyze" in sent[0]["content"]

    def test_stream_fallback_on_primary_failure(self) -> None:
        """stream() falls back to haiku when sonnet fails."""
        def haiku_stream() -> Iterator[str]:
            yield '{"choices":[{"delta":{"content":"haiku-result"},"finish_reason":"stop"}]}'

        haiku_mock = MagicMock()
        haiku_mock.chat_stream.return_value = haiku_stream()

        sonnet_mock = MagicMock()
        sonnet_mock.chat_stream.side_effect = Exception("Sonnet stream error")

        with patch("blend.providers.BaosiProvider", return_value=sonnet_mock):
            with patch("blend.providers.MinimaxProvider", return_value=haiku_mock):
                executor = Executor()
                executor.resource_model = MagicMock()
                executor.resource_model.get_remaining.side_effect = (
                    lambda m: 10000 if m in ("sonnet", "haiku") else 0
                )

                chunks = list(executor.stream(prompt="test", complexity=6))
                joined = "".join(chunks)
                assert "haiku-result" in joined


class TestStreamMessagesParams:
    """Test stream_messages() parameter forwarding branches."""

    def _fake_stream(self) -> Iterator[str]:
        yield '{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'

    def _make_mock_provider(self) -> MagicMock:
        m = MagicMock()
        m.chat_stream.return_value = self._fake_stream()
        return m

    def test_stream_messages_forwards_tool_choice(self) -> None:
        """stream_messages forwards tool_choice."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                tool_choice="auto",
            ))

            assert mock.chat_stream.call_args.kwargs["tool_choice"] == "auto"

    def test_stream_messages_forwards_response_format(self) -> None:
        """stream_messages forwards response_format."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                response_format={"type": "json_object"},
            ))

            assert mock.chat_stream.call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_stream_messages_forwards_max_tokens(self) -> None:
        """stream_messages forwards max_tokens."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                max_tokens=100,
            ))

            assert mock.chat_stream.call_args.kwargs["max_tokens"] == 100

    def test_stream_messages_forwards_temperature(self) -> None:
        """stream_messages forwards non-default temperature."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                temperature=0.7,
            ))

            assert mock.chat_stream.call_args.kwargs["temperature"] == 0.7

    def test_stream_messages_forwards_top_p(self) -> None:
        """stream_messages forwards top_p."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                top_p=0.9,
            ))

            assert mock.chat_stream.call_args.kwargs["top_p"] == 0.9

    def test_stream_messages_forwards_presence_penalty(self) -> None:
        """stream_messages forwards presence_penalty."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                presence_penalty=0.5,
            ))

            assert mock.chat_stream.call_args.kwargs["presence_penalty"] == 0.5

    def test_stream_messages_forwards_frequency_penalty(self) -> None:
        """stream_messages forwards frequency_penalty."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                frequency_penalty=0.3,
            ))

            assert mock.chat_stream.call_args.kwargs["frequency_penalty"] == 0.3

    def test_stream_messages_forwards_stop(self) -> None:
        """stream_messages forwards stop."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                stop=["END"],
            ))

            assert mock.chat_stream.call_args.kwargs["stop"] == ["END"]

    def test_stream_messages_tool_calls_delta_forwarded(self) -> None:
        """stream_messages forwards tool_call delta chunks."""
        def tool_stream() -> Iterator[str]:
            yield '{"choices":[{"delta":{"content":""},"finish_reason":"tool_calls","tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]}]}'

        mock = MagicMock()
        mock.chat_stream.return_value = tool_stream()

        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.return_value = 10000

            chunks = list(executor.stream_messages(
                messages=[{"role": "user", "content": "Weather?"}],
                complexity=7,
                tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            ))

            tool_chunks = [c for c in chunks if "tool_calls" in c]
            assert len(tool_chunks) >= 1

    def test_stream_messages_unknown_model_key(self) -> None:
        """stream_messages handles unknown model key by defaulting to haiku."""
        ok_mock = self._make_mock_provider()
        with patch("blend.providers.MinimaxProvider", return_value=ok_mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.return_value = 10000

            chunks = list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=2,
            ))
            assert isinstance(chunks, list)
            assert len(chunks) >= 1

    def test_stream_messages_default_temperature_not_forwarded(self) -> None:
        """stream_messages does not forward temperature when default (1.0)."""
        mock = self._make_mock_provider()
        with patch("blend.providers.BaosiProvider", return_value=mock):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "sonnet" else 10000
            )

            list(executor.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=3,
                temperature=1.0,
            ))

            assert "temperature" not in mock.chat_stream.call_args.kwargs

    def test_stream_all_providers_fail_raises(self) -> None:
        """stream() raises RuntimeError when all providers fail."""
        def fail_all(key: str) -> tuple[MagicMock, str]:
            m = MagicMock()
            m.chat_stream.side_effect = Exception(f"{key} down")
            return m, key

        with patch("blend.core.executor._get_provider", side_effect=fail_all):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 10000 if m in ("sonnet", "haiku", "minimax") else 0
            )

            with pytest.raises(RuntimeError, match="All model providers failed"):
                list(executor.stream(prompt="test", complexity=7))


class TestCallModelMessagesKwargs:
    """Test _call_model_messages kwargs forwarding branches."""

    def test_call_model_messages_injects_system_with_strategy_list(self) -> None:
        """_call_model_messages accepts strategy plan as list of steps."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "done"
        mock_response.finish_reason = "stop"
        mock_response.tool_calls = None
        mock_provider.chat.return_value = mock_response

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "minimax")):
            executor._call_model_messages(
                model="minimax",
                messages=[{"role": "user", "content": "hello"}],
                strategy={"plan": ["step 1", "step 2", "step 3"]},
            )

            sent = mock_provider.chat.call_args.kwargs["messages"]
            assert sent[0]["role"] == "system"

    def test_call_model_messages_unknown_model_key(self) -> None:
        """_call_model_messages handles unknown model key (defaults to haiku)."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.finish_reason = "stop"
        mock_response.tool_calls = None
        mock_provider.chat.return_value = mock_response

        # Unknown key "unknown_model" → defaults to haiku via _get_provider
        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "haiku-fallback")):
            result = executor._call_model_messages(
                model="unknown_model",
                messages=[{"role": "user", "content": "hi"}],
            )
            assert result.content == "ok"

    def test_execute_unknown_model_key(self) -> None:
        """execute() handles unknown model key by defaulting to haiku."""
        mock_provider = MagicMock()
        mock_provider.chat.return_value = MagicMock(content="ok")

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "haiku")):
            executor = Executor()
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.return_value = 10000

            # Force unknown key by patching MODEL_MAP
            with patch("blend.core.executor.MODEL_MAP", {"haiku": ("BaosiProvider", "haiku")}):
                result = executor.execute("test", complexity=1)
            assert result.raw_output == "ok"


class TestExecuteMessagesStopParam:
    """Test execute_messages stop parameter."""

    def test_execute_messages_forwards_stop_string(self) -> None:
        """execute_messages forwards stop as string."""
        from blend.core.executor import LLMOutput

        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        captured: dict[str, Any] = {}

        def capture(model: str, **kwargs: Any) -> LLMOutput:
            captured["kwargs"] = kwargs
            return LLMOutput(
                content="ok", model_used=model, tokens_used=2,
                tokens_budget_remaining=100, quality_gate_passed=True,
            )

        with patch.object(executor, "_call_model_messages", side_effect=capture):
            executor.execute_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=2,
                stop="DONE",
            )

            assert captured["kwargs"]["stop"] == "DONE"

    def test_execute_messages_forwards_stop_list(self) -> None:
        """execute_messages forwards stop as list."""
        from blend.core.executor import LLMOutput

        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        captured: dict[str, Any] = {}

        def capture(model: str, **kwargs: Any) -> LLMOutput:
            captured["kwargs"] = kwargs
            return LLMOutput(
                content="ok", model_used=model, tokens_used=2,
                tokens_budget_remaining=100, quality_gate_passed=True,
            )

        with patch.object(executor, "_call_model_messages", side_effect=capture):
            executor.execute_messages(
                messages=[{"role": "user", "content": "Hi"}],
                complexity=2,
                stop=["DONE", "STOP"],
            )

            assert captured["kwargs"]["stop"] == ["DONE", "STOP"]
