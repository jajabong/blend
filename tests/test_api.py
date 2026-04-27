"""Tests for story-009: API Service - FastAPI OpenAI Compatible."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from blend import __version__
from blend.api import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_endpoint(self) -> None:
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "blend"
        assert "status" in data  # healthy or degraded depending on provider reachability

    def test_root_endpoint(self) -> None:
        """Root endpoint should return service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "blend"
        assert "version" in data


class TestChatCompletions:
    """Test /v1/chat/completions endpoint."""

    def test_chat_completions_basic(self) -> None:
        """Basic chat completions should work."""
        mock_result = {
            "final_output": "Mock response",
            "layer_path": "L1>L3>L5",
            "complexity": 2,
            "model_used": "minimax",
            "tokens_used": 100,
            "quality_gate_passed": True,
            "l1_compressed": False,
            "l4_applied": False,
            "finish_reason": "stop",
            "tool_calls": None,
            "tool_call_count": 0,
            "tool_loop_iterations": 0,
        }
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), mock_result)()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_chat_completions_with_stream(self) -> None:
        """Streaming should be supported."""
        def mock_stream_messages(*args: object, **kwargs: object) -> object:
            yield {"id": "chatcmpl-test", "choices": [{"delta": {"content": "streamed"}, "finish_reason": None}], "_blend": {}}
            yield {"id": "chatcmpl-test", "choices": [{"delta": {}, "finish_reason": "stop"}], "_blend": {}}

        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.stream_messages.return_value = mock_stream_messages()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
        assert response.status_code == 200

    def test_chat_completions_response_format(self) -> None:
        """Response should match OpenAI format."""
        mock_result = {
            "final_output": "Test response",
            "layer_path": "L1>L3>L5",
            "complexity": 2,
            "model_used": "minimax",
            "tokens_used": 100,
            "quality_gate_passed": True,
            "l1_compressed": False,
            "l4_applied": False,
            "finish_reason": "stop",
            "tool_calls": None,
            "tool_call_count": 0,
            "tool_loop_iterations": 0,
        }
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), mock_result)()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        data = response.json()
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "model" in data
        assert "created" in data
        assert "choices" in data

    def test_chat_completions_empty_messages(self) -> None:
        """Empty messages list should return 400."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5",
                "messages": [],
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data


class TestModels:
    """Test /v1/models endpoint."""

    def test_list_models(self) -> None:
        """Should list available models."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0


class TestAPIIntegration:
    """Test API integration with blend layers."""

    def test_full_request_flow(self) -> None:
        """Test complete request through layers."""
        mock_result = {
            "final_output": "def add(a, b): return a + b",
            "layer_path": "L1>L3>L5",
            "complexity": 3,
            "model_used": "minimax",
            "tokens_used": 150,
            "quality_gate_passed": True,
            "l1_compressed": False,
            "l4_applied": False,
            "finish_reason": "stop",
            "tool_calls": None,
            "tool_call_count": 0,
            "tool_loop_iterations": 0,
        }
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), mock_result)()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [
                        {"role": "user", "content": "Write a Python function to add two numbers"}
                    ],
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data


class TestBudgetEndpoint:
    """Test /v1/budget endpoint."""

    def test_get_budget_returns_status(self) -> None:
        """Budget endpoint should return model budget status."""
        response = client.get("/v1/budget")
        assert response.status_code == 200
        data = response.json()
        assert "minimax" in data
        assert "haiku" in data
        assert "sonnet" in data
        assert "opus" in data
        for model_status in data.values():
            assert isinstance(model_status, dict)


class TestInfoEndpoint:
    """Test /v1/info endpoint."""

    def test_get_info_returns_system_info(self) -> None:
        """Info endpoint should return system information."""
        response = client.get("/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "blend"
        assert data["version"] == __version__
        assert "layer_architecture" in data
        assert "routing" in data
        assert "providers" in data
        providers = data["providers"]
        assert "minimax" in providers
        assert "baosi" in providers
        assert "lemon" in providers


class TestAPIExceptionHandling:
    """Test API exception handling."""

    def test_chat_completions_processing_error(self) -> None:
        """Processing errors should return 500 with detail."""
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.side_effect = RuntimeError("Internal error")
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Processing error" in data["detail"]

    def test_chat_completions_stream_content(self) -> None:
        """Streaming response should contain actual content chunks."""
        def mock_stream_messages(*args: object, **kwargs: object) -> object:
            yield {"id": "chatcmpl-test", "choices": [{"delta": {"content": "streamed content"}, "finish_reason": None}], "_blend": {}}
            yield {"id": "chatcmpl-test", "choices": [{"delta": {}, "finish_reason": "stop"}], "_blend": {}}

        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.stream_messages.return_value = mock_stream_messages()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        chunks = list(response.iter_lines())
        assert len(chunks) > 0
        # Last non-empty chunk should be [DONE]
        non_empty = [c for c in chunks if c]
        assert non_empty[-1] == "data: [DONE]"


class TestToolsPassthrough:
    """Test tools/tool_choice parameter passthrough."""

    def test_chat_with_tools(self) -> None:
        """tools parameter should be forwarded to orchestrator."""
        mock_result = {
            "final_output": "Tool call response",
            "layer_path": "L1>L3>L5",
            "complexity": 4,
            "model_used": "sonnet",
            "tokens_used": 80,
            "quality_gate_passed": True,
            "l1_compressed": False,
            "l4_applied": False,
            "finish_reason": "tool_calls",
            "tool_calls": None,
            "tool_call_count": 0,
            "tool_loop_iterations": 0,
        }
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), mock_result)()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Get the weather"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather for a city",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        },
                    ],
                },
            )
        assert response.status_code == 200
        mock_orch.process_messages.assert_called_once()
        call_kwargs = mock_orch.process_messages.call_args
        assert call_kwargs.kwargs.get("tools") is not None


class TestJSONMode:
    """Test JSON mode via response_format parameter."""

    def test_json_mode_response_format(self) -> None:
        """response_format parameter should be forwarded to orchestrator."""
        mock_result = {
            "final_output": '{"name": "Alice", "age": 30}',
            "layer_path": "L1>L3>L5",
            "complexity": 2,
            "model_used": "minimax",
            "tokens_used": 50,
            "quality_gate_passed": True,
            "l1_compressed": False,
            "l4_applied": False,
            "finish_reason": "stop",
            "tool_calls": None,
            "tool_call_count": 0,
            "tool_loop_iterations": 0,
        }
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), mock_result)()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Return JSON with name and age"}],
                    "response_format": {"type": "json_object"},
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        mock_orch.process_messages.assert_called_once()
        call_kwargs = mock_orch.process_messages.call_args
        assert call_kwargs.kwargs.get("response_format") == {"type": "json_object"}


class TestMultimodal:
    """Test multimodal content (image/audio) passthrough."""

    def test_multimodal_content_list(self) -> None:
        """content as list (multimodal) should be preserved through pipeline."""
        mock_result = {
            "final_output": "Image contains a cat",
            "layer_path": "L1>L3>L5",
            "complexity": 3,
            "model_used": "haiku",
            "tokens_used": 30,
            "quality_gate_passed": True,
            "l1_compressed": False,
            "l4_applied": False,
            "finish_reason": "stop",
            "tool_calls": None,
            "tool_call_count": 0,
            "tool_loop_iterations": 0,
        }
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), mock_result)()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/cat.png"},
                                },
                            ],
                        },
                    ],
                },
            )
        assert response.status_code == 200
        mock_orch.process_messages.assert_called_once()
        call_args = mock_orch.process_messages.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs.get("messages")
        assert messages is not None
        assert isinstance(messages[0]["content"], list)

    def test_multimodal_stream(self) -> None:
        """Streaming with multimodal content should work."""
        def mock_stream_messages(*args: object, **kwargs: object) -> object:
            yield {"id": "chatcmpl-test", "choices": [{"delta": {"content": "Image shows a sunset"}, "finish_reason": None}], "_blend": {}}
            yield {"id": "chatcmpl-test", "choices": [{"delta": {}, "finish_reason": "stop"}], "_blend": {}}

        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.stream_messages.return_value = mock_stream_messages()
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image"},
                                {"type": "image_url", "image_url": {"url": "https://example.com/sunset.jpg"}},
                            ],
                        },
                    ],
                    "stream": True,
                },
            )
        assert response.status_code == 200


class TestOpenAIExtendedParams:
    """Test extended OpenAI params (temperature, top_p, stop, etc.)."""

    def _mock_result(self, **overrides: object) -> dict[str, object]:
        base = {
            "final_output": "Hello",
            "layer_path": "L1>L3>L5",
            "complexity": 2,
            "model_used": "minimax",
            "tokens_used": 10,
            "quality_gate_passed": True,
            "l1_compressed": False,
            "l4_applied": False,
            "finish_reason": "stop",
            "tool_calls": None,
            "tool_call_count": 0,
            "tool_loop_iterations": 0,
        }
        base.update(overrides)
        return base

    def test_temperature_passed_to_orchestrator(self) -> None:
        """temperature should be forwarded to orchestrator."""
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), self._mock_result())()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "temperature": 0.7,
                },
            )
        call_kwargs = mock_orch.process_messages.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.7

    def test_top_p_passed_to_orchestrator(self) -> None:
        """top_p should be forwarded to orchestrator."""
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), self._mock_result())()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "top_p": 0.9,
                },
            )
        call_kwargs = mock_orch.process_messages.call_args.kwargs
        assert call_kwargs.get("top_p") == 0.9

    def test_stop_passed_to_orchestrator(self) -> None:
        """stop should be forwarded to orchestrator."""
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), self._mock_result())()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stop": "TERMINATE",
                },
            )
        call_kwargs = mock_orch.process_messages.call_args.kwargs
        assert call_kwargs.get("stop") == "TERMINATE"

    def test_stop_list_passed_to_orchestrator(self) -> None:
        """stop as list should be forwarded to orchestrator."""
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), self._mock_result())()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stop": ["TERMINATE", "END"],
                },
            )
        call_kwargs = mock_orch.process_messages.call_args.kwargs
        assert call_kwargs.get("stop") == ["TERMINATE", "END"]

    def test_max_tokens_passed_to_orchestrator(self) -> None:
        """max_tokens should be forwarded to orchestrator."""
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), self._mock_result())()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 500,
                },
            )
        call_kwargs = mock_orch.process_messages.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 500

    def test_presence_penalty_passed_to_orchestrator(self) -> None:
        """presence_penalty should be forwarded to orchestrator."""
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), self._mock_result())()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "presence_penalty": 0.5,
                },
            )
        call_kwargs = mock_orch.process_messages.call_args.kwargs
        assert call_kwargs.get("presence_penalty") == 0.5

    def test_frequency_penalty_passed_to_orchestrator(self) -> None:
        """frequency_penalty should be forwarded to orchestrator."""
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), self._mock_result())()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "frequency_penalty": 0.3,
                },
            )
        call_kwargs = mock_orch.process_messages.call_args.kwargs
        assert call_kwargs.get("frequency_penalty") == 0.3

    def test_stream_passes_extended_params(self) -> None:
        """Streaming mode should forward temperature/top_p/stop."""
        def mock_stream(*args: object, **kwargs: object) -> object:
            yield {"id": "chatcmpl-test", "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}], "_blend": {}}
            yield {"id": "chatcmpl-test", "choices": [{"delta": {}, "finish_reason": "stop"}], "_blend": {}}

        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.stream_messages.return_value = mock_stream()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                    "temperature": 0.5,
                    "stop": "DONE",
                },
            )
        call_kwargs = mock_orch.stream_messages.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("stop") == "DONE"


class TestAgentModeParam:
    """Test agent_mode parameter."""

    def test_agent_mode_passed_to_orchestrator(self) -> None:
        """agent_mode should be forwarded to orchestrator."""
        mock_result = {
            "final_output": "Done",
            "layer_path": "L1>L3",
            "complexity": 2,
            "model_used": "minimax",
            "tokens_used": 5,
            "quality_gate_passed": True,
            "l1_compressed": False,
            "l4_applied": False,
            "finish_reason": "stop",
            "tool_calls": None,
            "tool_call_count": 0,
            "tool_loop_iterations": 0,
        }
        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.process_messages.return_value = type("MockResult", (), mock_result)()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "agent_mode": True,
                },
            )
        call_kwargs = mock_orch.process_messages.call_args.kwargs
        assert call_kwargs.get("agent_mode") is True

    def test_agent_mode_stream_passes_through(self) -> None:
        """agent_mode should be forwarded in streaming mode."""
        def mock_stream(*args: object, **kwargs: object) -> object:
            yield {"id": "chatcmpl-test", "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}], "_blend": {}}
            yield {"id": "chatcmpl-test", "choices": [{"delta": {}, "finish_reason": "stop"}], "_blend": {}}

        with patch("blend.api.orchestrator") as mock_orch:
            mock_orch.stream_messages.return_value = mock_stream()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                    "agent_mode": True,
                },
            )
        call_kwargs = mock_orch.stream_messages.call_args.kwargs
        assert call_kwargs.get("agent_mode") is True

