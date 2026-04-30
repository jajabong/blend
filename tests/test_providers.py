"""Tests for all LLM providers."""

from unittest.mock import MagicMock, patch

import pytest

from blend.providers.baosiapi import BaosiProvider
from blend.providers.lemonapi import LemonProvider
from blend.providers.minimax import MinimaxProvider


def _mock_circuit_breaker():
    """Return a mock circuit breaker that allows all requests."""
    mock_breaker = MagicMock()
    mock_breaker.allow_request.return_value = True
    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_breaker
    return mock_registry


class TestLemonProvider:
    """Tests for LemonProvider (Gemini)."""

    def test_chat_success(self) -> None:
        """chat() returns LemonResponse on success."""
        mock_data = {
            "choices": [{"message": {"content": "Gemini response"}}],
            "model": "[L]gemini-3-flash-preview",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        with patch("blend.providers.lemonapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = LemonProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                result = provider.chat(messages=[{"role": "user", "content": "hello"}])

            assert result.content == "Gemini response"
            assert result.model == "[L]gemini-3-flash-preview"
            assert result.usage["prompt_tokens"] == 10

    def test_chat_uses_default_model(self) -> None:
        """chat() uses DEFAULT_MODEL when model not specified."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "[L]gemini-3-flash-preview",
            "usage": {},
        }
        with patch("blend.providers.lemonapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = LemonProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                provider.chat(messages=[{"role": "user", "content": "hi"}])

            call_kwargs = mock_client.post.call_args[1]
            payload = call_kwargs["json"]
            assert payload["model"] == "[L]gemini-3-flash-preview"
            assert payload["stream"] is False

    def test_chat_custom_model(self) -> None:
        """chat() uses custom model when specified."""
        mock_data = {
            "choices": [{"message": {"content": "pro response"}}],
            "model": "[L]gemini-3-pro-preview",
            "usage": {},
        }
        with patch("blend.providers.lemonapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = LemonProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                result = provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="[L]gemini-3-pro-preview",
                )

            call_kwargs = mock_client.post.call_args[1]
            assert call_kwargs["json"]["model"] == "[L]gemini-3-pro-preview"
            assert result.model == "[L]gemini-3-pro-preview"

    def test_chat_includes_extra_kwargs(self) -> None:
        """chat() passes through extra kwargs."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "[L]gemini-3-flash-preview",
            "usage": {},
        }
        with patch("blend.providers.lemonapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = LemonProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.7,
                    max_tokens=500,
                )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["temperature"] == 0.7
            assert payload["max_tokens"] == 500

    def test_chat_raises_on_http_error(self) -> None:
        """chat() propagates httpx HTTP errors."""
        import httpx

        with patch("blend.providers.lemonapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = LemonProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPError("server error")
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                with pytest.raises(httpx.HTTPError):
                    provider.chat(messages=[{"role": "user", "content": "hi"}])

    def test_chat_uses_env_api_key(self) -> None:
        """Uses LEMON_API_KEY from environment when no key provided."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "[L]gemini-3-flash-preview",
            "usage": {},
        }
        with patch("blend.providers.lemonapi.get_registry", return_value=_mock_circuit_breaker()):
            with patch.dict("os.environ", {"LEMON_API_KEY": "env-key"}):
                provider = LemonProvider()
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.json.return_value = mock_data
                mock_client.post.return_value = mock_response
                with patch.object(provider, "_get_client", return_value=mock_client):
                    provider.chat(messages=[{"role": "user", "content": "hi"}])

                headers = mock_client.post.call_args[1]["headers"]
                assert "env-key" in headers["Authorization"]

    def test_chat_stream_returns_chunks(self) -> None:
        """chat_stream() returns parsed chunks."""
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]

        with patch("blend.providers.lemonapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = LemonProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_stream_response = MagicMock()
            mock_stream_response.status_code = 200
            mock_stream_response.raise_for_status = MagicMock()
            mock_stream_response.iter_lines.return_value = iter(mock_lines)
            mock_client.stream.return_value.__enter__.return_value = mock_stream_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                chunks = provider.chat_stream(
                    messages=[{"role": "user", "content": "hi"}]
                )

            # [DONE] line triggers break before last content chunk is processed,
            # so only the chunks before [DONE] are returned
            assert len(chunks) == 2
            assert 'content":"hello"' in chunks[0]
            assert mock_stream_response.raise_for_status.called

    def test_chat_stream_breaks_on_done(self) -> None:
        """chat_stream() stops at [DONE]."""
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"should not appear"}}]}',
        ]

        with patch("blend.providers.lemonapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = LemonProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_stream_response = MagicMock()
            mock_stream_response.status_code = 200
            mock_stream_response.raise_for_status = MagicMock()
            mock_stream_response.iter_lines.return_value = iter(mock_lines)
            mock_client.stream.return_value.__enter__.return_value = mock_stream_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                chunks = provider.chat_stream(messages=[{"role": "user", "content": "hi"}])

            # [DONE] stops loop; "should not appear" never reached
            assert len(chunks) == 2
            assert 'content":"a"' in chunks[0]


class TestBaosiProvider:
    """Tests for BaosiProvider (Claude)."""

    def test_chat_success(self) -> None:
        """chat() returns BaosiResponse on success."""
        mock_data = {
            "choices": [{"message": {"content": "Claude response"}}],
            "model": "claude-sonnet-4-6",
            "usage": {"prompt_tokens": 5, "completion_tokens": 15},
        }
        with patch("blend.providers.baosiapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = BaosiProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                result = provider.chat(
                    messages=[{"role": "user", "content": "hello"}],
                    model="claude-sonnet-4-6",
                )

            assert result.content == "Claude response"
            assert result.model == "claude-sonnet-4-6"
            assert result.usage["completion_tokens"] == 15

    def test_chat_default_model(self) -> None:
        """chat() uses provided model param."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "claude-opus-4-7",
            "usage": {},
        }
        with patch("blend.providers.baosiapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = BaosiProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-opus-4-7",
                )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["model"] == "claude-opus-4-7"

    def test_chat_includes_kwargs(self) -> None:
        """chat() passes through temperature, max_tokens."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "claude-sonnet-4-6",
            "usage": {},
        }
        with patch("blend.providers.baosiapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = BaosiProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.5,
                    max_tokens=1024,
                )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["temperature"] == 0.5
            assert payload["max_tokens"] == 1024

    def test_chat_raises_on_http_error(self) -> None:
        """chat() propagates HTTP errors."""
        import httpx

        with patch("blend.providers.baosiapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = BaosiProvider(api_key="bad-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPError("403 Forbidden")
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                with pytest.raises(httpx.HTTPError):
                    provider.chat(messages=[{"role": "user", "content": "hi"}])

    def test_chat_stream(self) -> None:
        """chat_stream() returns parsed chunks."""
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"step1"}}]}',
            "data: [DONE]",
        ]

        with patch("blend.providers.baosiapi.get_registry", return_value=_mock_circuit_breaker()):
            provider = BaosiProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_stream_response = MagicMock()
            mock_stream_response.status_code = 200
            mock_stream_response.raise_for_status = MagicMock()
            mock_stream_response.iter_lines.return_value = iter(mock_lines)
            mock_client.stream.return_value.__enter__.return_value = mock_stream_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                chunks = list(provider.chat_stream(messages=[{"role": "user", "content": "hi"}]))

            assert len(chunks) == 1
            assert 'content":"step1"' in chunks[0]

    def test_list_models(self) -> None:
        """list_models() returns list of model dicts."""
        mock_data = {
            "data": [
                {"id": "claude-opus-4-7", "object": "model"},
                {"id": "claude-sonnet-4-6", "object": "model"},
            ]
        }

        provider = BaosiProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_client.get.return_value = mock_response
        with patch.object(provider, "_get_client", return_value=mock_client):
            models = provider.list_models()

            assert len(models) == 2
            assert models[0]["id"] == "claude-opus-4-7"
            mock_client.get.assert_called_once()

    def test_list_models_http_error(self) -> None:
        """list_models() propagates HTTP errors."""
        import httpx

        provider = BaosiProvider(api_key="bad-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError("401 Unauthorized")
        mock_client.get.return_value = mock_response
        with patch.object(provider, "_get_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPError):
                provider.list_models()


class TestMinimaxProvider:
    """Tests for MinimaxProvider."""

    def test_chat_success(self) -> None:
        """chat() returns MinimaxResponse on success."""
        mock_data = {
            "choices": [{"message": {"content": "Minimax response"}}],
            "model": "MiniMax-M2.7",
            "usage": {"prompt_tokens": 8, "completion_tokens": 12},
        }
        with patch("blend.providers.minimax.get_registry", return_value=_mock_circuit_breaker()):
            provider = MinimaxProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                result = provider.chat(messages=[{"role": "user", "content": "hello"}])

            assert result.content == "Minimax response"
            assert result.model == "MiniMax-M2.7"
            assert result.usage["completion_tokens"] == 12

    def test_chat_default_model(self) -> None:
        """chat() uses MiniMax-M2.7 when model not specified."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "MiniMax-M2.7",
            "usage": {},
        }
        with patch("blend.providers.minimax.get_registry", return_value=_mock_circuit_breaker()):
            provider = MinimaxProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                provider.chat(messages=[{"role": "user", "content": "hi"}])

            payload = mock_client.post.call_args[1]["json"]
            assert payload["model"] == "MiniMax-M2.7"
            assert payload["stream"] is False

    def test_chat_custom_model(self) -> None:
        """chat() uses custom model when specified."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "other-model",
            "usage": {},
        }
        with patch("blend.providers.minimax.get_registry", return_value=_mock_circuit_breaker()):
            provider = MinimaxProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                result = provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="other-model",
                )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["model"] == "other-model"
            assert result.model == "other-model"

    def test_chat_includes_extra_kwargs(self) -> None:
        """chat() passes through extra kwargs."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "MiniMax-M2.7",
            "usage": {},
        }
        with patch("blend.providers.minimax.get_registry", return_value=_mock_circuit_breaker()):
            provider = MinimaxProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.9,
                )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["temperature"] == 0.9

    def test_chat_raises_on_http_error(self) -> None:
        """chat() propagates HTTP errors."""
        import httpx

        with patch("blend.providers.minimax.get_registry", return_value=_mock_circuit_breaker()):
            provider = MinimaxProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPError("429 Rate limit")
            mock_client.post.return_value = mock_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                with pytest.raises(httpx.HTTPError):
                    provider.chat(messages=[{"role": "user", "content": "hi"}])

    def test_chat_uses_env_api_key(self) -> None:
        """Uses MINIMAX_API_KEY from environment."""
        mock_data = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "MiniMax-M2.7",
            "usage": {},
        }
        with patch("blend.providers.minimax.get_registry", return_value=_mock_circuit_breaker()):
            with patch.dict("os.environ", {"MINIMAX_API_KEY": "env-key"}):
                provider = MinimaxProvider()
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.json.return_value = mock_data
                mock_client.post.return_value = mock_response
                with patch.object(provider, "_get_client", return_value=mock_client):
                    provider.chat(messages=[{"role": "user", "content": "hi"}])

                headers = mock_client.post.call_args[1]["headers"]
                assert "env-key" in headers["Authorization"]

    def test_chat_stream(self) -> None:
        """chat_stream() returns parsed chunks."""
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"chunk1"}}]}',
            "data: [DONE]",
        ]

        with patch("blend.providers.minimax.get_registry", return_value=_mock_circuit_breaker()):
            provider = MinimaxProvider(api_key="test-key")
            mock_client = MagicMock()
            mock_stream_response = MagicMock()
            mock_stream_response.status_code = 200
            mock_stream_response.raise_for_status = MagicMock()
            mock_stream_response.iter_lines.return_value = iter(mock_lines)
            mock_client.stream.return_value.__enter__.return_value = mock_stream_response
            with patch.object(provider, "_get_client", return_value=mock_client):
                chunks = list(provider.chat_stream(messages=[{"role": "user", "content": "hi"}]))

            assert len(chunks) == 1
            assert 'content":"chunk1"' in chunks[0]
            mock_stream_response.raise_for_status.assert_called_once()
