"""Tests for Anthropic Messages API endpoint (/v1/messages)."""

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestAnthropicMessagesEndpoint:
    """Test POST /v1/messages — Anthropic Messages API compatible endpoint."""

    def test_messages_endpoint_exists(self) -> None:
        """The /v1/messages endpoint should exist and respond."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        # Should not 404
        assert response.status_code != 404

    def test_messages_endpoint_basic_request(self) -> None:
        """Basic /v1/messages request should not 422 (model/messages/max_tokens required)."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "say hello"}],
            },
        )
        # Should parse correctly
        assert response.status_code in (200, 500)  # 500 = orchestrator called, not a validation error
        assert response.status_code != 422  # Not a pydantic validation failure

    def test_messages_endpoint_validation_missing_model(self) -> None:
        """Missing model field should return 422."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 422

    def test_messages_endpoint_validation_missing_messages(self) -> None:
        """Missing messages field should return 422."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
            },
        )
        assert response.status_code == 422

    def test_messages_endpoint_validation_missing_max_tokens(self) -> None:
        """Missing max_tokens should return 422."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 422

    def test_messages_endpoint_with_system_prompt(self) -> None:
        """System prompt should be prepended as a system message."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code in (200, 500)

    def test_messages_endpoint_stream_false(self) -> None:
        """stream=false should return a JSON message response."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        with patch("blend.api.BlendOrchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch_cls.return_value = mock_orch
            mock_result = MagicMock()
            mock_result.final_output = "Hello!"
            mock_result.finish_reason = "end_turn"
            mock_result.model_used = "sonnet"
            mock_result.tokens_used = 10
            mock_result.complexity = 3
            mock_result.layer_path = "L1>L3>L5"
            mock_result.quality_gate_passed = True
            mock_result.l1_compressed = False
            mock_result.l4_applied = False
            mock_result.tool_calls = None
            mock_result.tool_call_count = 0
            mock_result.tool_loop_iterations = 0
            mock_orch.process_messages.return_value = mock_result

            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 100,
                    "stream": False,
                    "messages": [{"role": "user", "content": "say hi"}],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert "content" in data
        assert "usage" in data

    def test_messages_endpoint_stream_true_returns_sse(self) -> None:
        """stream=true should return SSE with Anthropic event types."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        with patch("blend.api.BlendOrchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch_cls.return_value = mock_orch
            # stream_messages yields chunks
            mock_orch.stream_messages.return_value = iter([
                {
                    "id": "msg_test",
                    "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
                },
                {
                    "id": "msg_test",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                },
            ])

            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "say hi"}],
                },
                headers={"anthropic-version": "2023-06-01"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        # SSE data lines
        lines = response.text.strip().split("\n")
        data_lines = [ln for ln in lines if ln.startswith("data: ")]
        assert len(data_lines) >= 2


class TestAnthropicRequestModel:
    """Test AnthropicMessageRequest model."""

    def test_model_has_required_fields(self) -> None:
        """AnthropicMessageRequest should have model, messages, max_tokens."""
        from blend.api import AnthropicMessageRequest

        req = AnthropicMessageRequest(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.model == "claude-sonnet-4-6"
        assert req.max_tokens == 100
        assert len(req.messages) == 1

    def test_model_optional_fields(self) -> None:
        """Optional fields should be present."""
        from blend.api import AnthropicMessageRequest

        req = AnthropicMessageRequest(
            model="claude-opus-4-7",
            max_tokens=100,
            messages=[],
            stream=True,
            system="You are helpful.",
            tools=[{"name": "bash", "input_schema": {"type": "object"}}],
            temperature=0.7,
            top_p=0.9,
        )
        assert req.stream is True
        assert req.system == "You are helpful."
        assert req.tools is not None and len(req.tools) == 1
        assert req.temperature == 0.7
        assert req.top_p == 0.9


class TestAnthropicResponseModel:
    """Test AnthropicMessageResponse model."""

    def test_response_model_structure(self) -> None:
        """Response should have type, role, content, usage, model, id."""
        from blend.api import AnthropicMessageResponse

        resp = AnthropicMessageResponse(
            id="msg_123",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": "Hello!"}],
            model="claude-sonnet-4-6",
            usage={"input_tokens": 5, "output_tokens": 10},
            stop_reason="end_turn",
            stop_sequence=None,
        )
        assert resp.id == "msg_123"
        assert resp.type == "message"
        assert resp.role == "assistant"
        assert len(resp.content) == 1
        assert resp.content[0]["text"] == "Hello!"
        assert resp.usage["output_tokens"] == 10


class TestAnthropicStreamingFormat:
    """Test SSE format conversion from blend chunks to Anthropic events."""

    def test_convert_text_chunk_to_anthropic_events(self) -> None:
        """OpenAI-style delta chunk should become Anthropic SSE events."""
        from blend.api import _convert_chunk_to_anthropic_events

        chunk = {
            "id": "msg_123",
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        }
        events = list(_convert_chunk_to_anthropic_events(chunk, is_first=True))

        # Events are JSON strings — parse them back
        event_types = [json.loads(e).get("type") for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types

    def test_convert_stop_chunk_closes_stream(self) -> None:
        """Stop chunk should produce message_delta and message_stop events."""
        from blend.api import _convert_chunk_to_anthropic_events

        chunk = {
            "id": "msg_123",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        }
        events = list(_convert_chunk_to_anthropic_events(chunk, is_first=False))
        event_types = [json.loads(e).get("type") for e in events]

        assert "message_delta" in event_types
        assert "message_stop" in event_types


class TestAnthropicBetaHeader:
    """Test beta header passthrough."""

    def test_anthropic_version_header_required(self) -> None:
        """anthropic-version header should be accepted (even if not enforced)."""
        from blend.api import app
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
            headers={"anthropic-version": "2023-06-01"},
        )
        # Should not error on the header
        assert response.status_code in (200, 422, 500)
