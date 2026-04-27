"""Comprehensive tests for Executor execute_messages method."""

from unittest.mock import MagicMock, patch

from blend.core.executor import Executor


class TestExecuteMessages:
    """Test execute_messages method."""

    def test_execute_messages_basic(self) -> None:
        """Basic execute_messages call works."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello world"
        mock_response.finish_reason = "stop"
        mock_response.tool_calls = None
        mock_provider.chat.return_value = mock_response

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "sonnet")):
            result = executor.execute_messages(
                messages=[{"role": "user", "content": "Hello"}],
                complexity=5,
            )

        assert result.content == "Hello world"
        assert result.finish_reason == "stop"

    def test_execute_messages_with_tools(self) -> None:
        """execute_messages with tools parameter."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Using tool"
        mock_response.finish_reason = "tool_calls"
        mock_response.tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        mock_provider.chat.return_value = mock_response

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "sonnet")):
            tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
            result = executor.execute_messages(
                messages=[{"role": "user", "content": "Use tool"}],
                complexity=5,
                tools=tools,
            )

        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None

    def test_execute_messages_with_temperature(self) -> None:
        """execute_messages passes temperature parameter."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Result"
        mock_response.finish_reason = "stop"
        mock_response.tool_calls = None
        mock_provider.chat.return_value = mock_response

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "sonnet")):
            executor.execute_messages(
                messages=[{"role": "user", "content": "Test"}],
                complexity=5,
                temperature=0.7,
            )

        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.7

    def test_execute_messages_with_max_tokens(self) -> None:
        """execute_messages passes max_tokens parameter."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Result"
        mock_response.finish_reason = "stop"
        mock_response.tool_calls = None
        mock_provider.chat.return_value = mock_response

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "sonnet")):
            executor.execute_messages(
                messages=[{"role": "user", "content": "Test"}],
                complexity=5,
                max_tokens=100,
            )

        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 100

    def test_execute_messages_with_stop(self) -> None:
        """execute_messages passes stop parameter."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Result"
        mock_response.finish_reason = "stop"
        mock_response.tool_calls = None
        mock_provider.chat.return_value = mock_response

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "sonnet")):
            executor.execute_messages(
                messages=[{"role": "user", "content": "Test"}],
                complexity=5,
                stop=["END"],
            )

        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs.get("stop") == ["END"]

    def test_execute_messages_fallback_on_exception(self) -> None:
        """execute_messages falls back to haiku when sonnet fails."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_sonnet = MagicMock()
        mock_sonnet.chat.side_effect = Exception("Sonnet down")

        mock_haiku = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Fallback result"
        mock_response.finish_reason = "stop"
        mock_response.tool_calls = None
        mock_haiku.chat.return_value = mock_response

        def provider_side_effect(key: str):
            if key == "sonnet":
                return mock_sonnet, "sonnet"
            return mock_haiku, "haiku"

        with patch("blend.core.executor._get_provider", side_effect=provider_side_effect):
            result = executor.execute_messages(
                messages=[{"role": "user", "content": "Test"}],
                complexity=5,
            )

        assert result.content == "Fallback result"
        assert result.model_used == "haiku"

    def test_execute_messages_with_response_format(self) -> None:
        """execute_messages passes response_format parameter."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "JSON result"
        mock_response.finish_reason = "stop"
        mock_response.tool_calls = None
        mock_provider.chat.return_value = mock_response

        with patch("blend.core.executor._get_provider", return_value=(mock_provider, "sonnet")):
            executor.execute_messages(
                messages=[{"role": "user", "content": "Give JSON"}],
                complexity=5,
                response_format={"type": "json_object"},
            )

        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}


class TestExecutorStream:
    """Test stream method."""

    def test_stream_basic(self) -> None:
        """Basic streaming works."""
        from blend.core.executor import Executor, ModelSelection

        executor = Executor()
        executor.resource_model = MagicMock()

        mock_provider = MagicMock()
        mock_provider.chat_stream.return_value = iter([
            '{"choices": [{"delta": {"content": "Hello"}}]}',
            '{"choices": [{"delta": {"content": " world"}}]}',
        ])

        with patch.object(executor, "_select_model", return_value=ModelSelection(primary="sonnet", fallback=[])):
            with patch("blend.core.executor._get_provider", return_value=(mock_provider, "sonnet")):
                chunks = list(executor.stream(
                    prompt="Say hello",
                    complexity=5,
                ))

        assert len(chunks) == 2
        assert chunks[0] == "Hello"

    def test_stream_with_strategy(self) -> None:
        """Stream passes strategy to provider."""
        from blend.core.executor import Executor, ModelSelection

        executor = Executor()
        executor.resource_model = MagicMock()

        mock_provider = MagicMock()
        mock_provider.chat_stream.return_value = iter([
            '{"choices": [{"delta": {"content": "Result"}}]}',
        ])

        with patch.object(executor, "_select_model", return_value=ModelSelection(primary="sonnet", fallback=[])):
            with patch("blend.core.executor._get_provider", return_value=(mock_provider, "sonnet")):
                list(executor.stream(
                    prompt="Test",
                    complexity=8,
                    strategy={"plan": ["Step 1", "Step 2"]},
                ))

        # Verify chat_stream was called
        mock_provider.chat_stream.assert_called_once()

    def test_stream_all_providers_fail(self) -> None:
        """Stream raises when all providers fail."""
        executor = Executor()
        executor.resource_model = MagicMock()
        executor.resource_model.get_remaining.return_value = 10000

        def provider_side_effect(key: str):
            mock = MagicMock()
            mock.chat_stream.side_effect = Exception("Provider down")
            return mock, key

        with patch("blend.core.executor._get_provider", side_effect=provider_side_effect):
            try:
                list(executor.stream(prompt="Test", complexity=5))
                assert False, "Should have raised"
            except RuntimeError as e:
                assert "failed" in str(e).lower()
