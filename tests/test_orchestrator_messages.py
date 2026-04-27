"""Tests for orchestrator message-based pipeline (tools, multimodal, JSON mode)."""

from unittest.mock import MagicMock, patch

from blend.core.orchestrator import BlendOrchestrator, OrchestratorResult


class TestProcessMessages:
    """Test process_messages with tools/response_format passthrough."""

    def test_process_messages_returns_result_with_tool_fields(self) -> None:
        """process_messages should return OrchestratorResult with finish_reason/tool_calls."""
        orchestrator = BlendOrchestrator()
        with patch.object(orchestrator.scorer, "score") as mock_score, \
             patch.object(orchestrator.executor, "execute_messages") as mock_exec, \
             patch.object(orchestrator.compression_trigger, "should_compress", return_value=False), \
             patch.object(orchestrator.verifier, "verify") as mock_verify, \
             patch.object(orchestrator.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(
                total=3, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_exec.return_value = MagicMock(
                content="test output",
                model_used="haiku",
                tokens_used=50,
                quality_gate_passed=True,
                finish_reason="tool_calls",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
            )
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            result = orchestrator.process_messages(
                messages=[{"role": "user", "content": "Get weather for Tokyo"}],
                tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
                tool_choice="auto",
                response_format={"type": "json_object"},
            )

            assert isinstance(result, OrchestratorResult)
            assert result.finish_reason == "tool_calls"
            assert result.tool_calls is not None
            assert result.tool_calls[0]["function"]["name"] == "get_weather"
            assert result.final_output == "test output"

    def test_process_messages_forwards_tools_to_executor(self) -> None:
        """tools and response_format should be forwarded to executor."""
        orchestrator = BlendOrchestrator()
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        response_format = {"type": "json_object"}

        with patch.object(orchestrator.scorer, "score") as mock_score, \
             patch.object(orchestrator.executor, "execute_messages") as mock_exec, \
             patch.object(orchestrator.compression_trigger, "should_compress", return_value=False), \
             patch.object(orchestrator.verifier, "verify") as mock_verify, \
             patch.object(orchestrator.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(
                total=2, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_exec.return_value = MagicMock(
                content="result", model_used="minimax", tokens_used=10,
                quality_gate_passed=True, finish_reason="stop", tool_calls=None,
            )
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            orchestrator.process_messages(
                messages=[{"role": "user", "content": "Hello"}],
                tools=tools,
                response_format=response_format,
            )

            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["tools"] == tools
            assert call_kwargs["response_format"] == response_format

    def test_process_messages_with_multimodal_content(self) -> None:
        """Multimodal content (list) should be preserved through scoring."""
        orchestrator = BlendOrchestrator()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
                ],
            }
        ]

        with patch.object(orchestrator.scorer, "score") as mock_score, \
             patch.object(orchestrator.executor, "execute_messages") as mock_exec, \
             patch.object(orchestrator.compression_trigger, "should_compress", return_value=False), \
             patch.object(orchestrator.verifier, "verify") as mock_verify, \
             patch.object(orchestrator.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(
                total=3, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_exec.return_value = MagicMock(
                content="Image shows a cat",
                model_used="haiku", tokens_used=20,
                quality_gate_passed=True, finish_reason="stop", tool_calls=None,
            )
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            orchestrator.process_messages(messages=messages)

            # Verify executor received messages (preserved structure)
            mock_exec.assert_called_once()
            exec_messages = mock_exec.call_args.kwargs["messages"]
            assert isinstance(exec_messages[0]["content"], list)


class TestMessagesToPrompt:
    """Test _messages_to_prompt conversion utility."""

    def test_text_messages_converted_to_prompt(self) -> None:
        """Simple text messages should be joined into prompt."""
        orchestrator = BlendOrchestrator()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        prompt = orchestrator._messages_to_prompt(messages)
        assert "system: You are helpful." in prompt
        assert "user: What is 2+2?" in prompt

    def test_multimodal_content_marked_as_media(self) -> None:
        """Multimodal content should be marked as [role: media content]."""
        orchestrator = BlendOrchestrator()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        prompt = orchestrator._messages_to_prompt(messages)
        assert "user: Describe this image" in prompt
        assert "[user: media content]" in prompt

    def test_tool_result_message_preserved(self) -> None:
        """Tool result messages should be preserved by role/content."""
        orchestrator = BlendOrchestrator()
        messages = [
            {"role": "user", "content": "Get weather"},
            {"role": "tool", "tool_call_id": "call_abc", "content": '{"temp": 22}'},
        ]
        prompt = orchestrator._messages_to_prompt(messages)
        assert "user: Get weather" in prompt
        assert "tool: " in prompt
        assert "call_abc" not in prompt  # only role+content preserved


class TestOrchestratorResultFields:
    """Test OrchestratorResult has all required fields."""

    def test_orchestrator_result_with_all_fields(self) -> None:
        """OrchestratorResult should accept finish_reason and tool_calls."""
        result = OrchestratorResult(
            final_output="test",
            layer_path="L1>L3>L5",
            complexity=3,
            model_used="minimax",
            tokens_used=100,
            quality_gate_passed=True,
            l1_compressed=False,
            l4_applied=False,
            finish_reason="tool_calls",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
        )
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "foo"

    def test_orchestrator_result_default_finish_reason(self) -> None:
        """finish_reason defaults to 'stop' when not provided."""
        result = OrchestratorResult(
            final_output="hello",
            layer_path="L1>L3>L5",
            complexity=1,
            model_used="minimax",
            tokens_used=10,
            quality_gate_passed=True,
            l1_compressed=False,
            l4_applied=False,
        )
        assert result.finish_reason == "stop"
        assert result.tool_calls is None


class TestStreamMessages:
    """Test stream_messages with agent_mode passthrough."""

    def test_stream_messages_yields_chunks(self) -> None:
        """stream_messages yields chunks from executor."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        with patch.object(orch.scorer, "score") as mock_score, \
             patch.object(orch.executor, "stream_messages") as mock_stream:

            mock_score.return_value = MagicMock(
                total=2, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )

            mock_stream.return_value = iter([
                {"delta": {"content": "Hello"}, "finish_reason": None},
            ])

            chunks = list(orch.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
            ))

            # Executor yields 1 content chunk; orchestrator adds 1 terminal stop chunk → 2 total
            assert len(chunks) == 2
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"

    def test_stream_messages_passes_agent_mode(self) -> None:
        """stream_messages passes agent_mode to executor.stream_messages."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        with patch.object(orch.scorer, "score") as mock_score, \
             patch.object(orch.executor, "stream_messages") as mock_stream:

            mock_score.return_value = MagicMock(
                total=2, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_stream.return_value = iter([{"delta": {"content": "Hi"}, "finish_reason": None}])

            list(orch.stream_messages(
                messages=[{"role": "user", "content": "Hi"}],
                agent_mode=True,
            ))

            # agent_mode=True must be forwarded to executor
            mock_stream.assert_called_once()
            call_kwargs = mock_stream.call_args.kwargs
            assert call_kwargs["agent_mode"] is True

    def test_stream_messages_forwards_tools(self) -> None:
        """stream_messages forwards tools to executor."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        with patch.object(orch.scorer, "score") as mock_score, \
             patch.object(orch.executor, "stream_messages") as mock_stream:

            mock_score.return_value = MagicMock(
                total=4, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_stream.return_value = iter([{"delta": {"content": "Calling..."}, "finish_reason": None}])

            list(orch.stream_messages(
                messages=[{"role": "user", "content": "Weather?"}],
                tools=tools,
            ))

            mock_stream.assert_called_once()
            assert mock_stream.call_args.kwargs["tools"] == tools

    def test_stream_messages_includes_blend_metadata(self) -> None:
        """stream_messages includes _blend metadata in each chunk."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        with patch.object(orch.scorer, "score") as mock_score, \
             patch.object(orch.executor, "stream_messages") as mock_stream:

            mock_score.return_value = MagicMock(
                total=3, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_stream.return_value = iter([
                {"delta": {"content": "Hi"}, "finish_reason": None},
            ])

            chunks = list(orch.stream_messages(
                messages=[{"role": "user", "content": "Hello"}],
            ))

            for chunk in chunks:
                assert "_blend" in chunk
                assert "complexity" in chunk["_blend"]
                assert "layer_path" in chunk["_blend"]
