"""Tests for v1.5 agent_mode, MCP tools, and context budget management."""

from typing import Any
from unittest.mock import MagicMock


class TestAgentModeInOrchestrator:
    """Test agent_mode flag bypasses L4/L5 in process_messages."""

    def test_agent_mode_disables_l4_compression(self) -> None:
        """L4 should NOT be triggered when agent_mode=True."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        orch.resource_model = MagicMock()
        orch.resource_model.track_consumption = MagicMock()

        orch.compression_trigger = MagicMock()
        orch.compression_trigger.should_compress.return_value = False
        orch.l4_compressor = MagicMock()
        orch.l4_compressor.compress.return_value = MagicMock(
            compressed_output="compressed",
            original_tokens=600,
            compressed_tokens=300,
            compression_ratio=0.5,
        )

        mock_result = MagicMock()
        mock_result.finish_reason = "stop"
        mock_result.tool_calls = None
        mock_result.content = "a" * 600
        mock_result.model_used = "minimax"
        mock_result.tokens_used = 600
        orch.executor = MagicMock()
        orch.executor.execute_messages.return_value = mock_result

        orch.scorer = MagicMock()
        orch.scorer.score.return_value = MagicMock(
            total=3, tier="LOW", task_type="general"
        )

        result = orch.process_messages(
            messages=[{"role": "user", "content": "hello"}],
            agent_mode=True,
        )

        orch.l4_compressor.compress.assert_not_called()
        assert result.l4_applied is False
        assert "L4" not in result.layer_path

    def test_agent_mode_skips_p0_check(self) -> None:
        """L5 P0 check should be bypassed when agent_mode=True."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        orch.resource_model = MagicMock()
        orch.resource_model.track_consumption = MagicMock()

        mock_result = MagicMock()
        mock_result.finish_reason = "stop"
        mock_result.tool_calls = None
        mock_result.content = "result = eval('2+2')  # dangerous code"
        mock_result.model_used = "minimax"
        mock_result.tokens_used = 20
        orch.executor = MagicMock()
        orch.executor.execute_messages.return_value = mock_result

        orch.scorer = MagicMock()
        orch.scorer.score.return_value = MagicMock(
            total=3, tier="LOW", task_type="general"
        )

        orch.compression_trigger = MagicMock()
        orch.compression_trigger.should_compress.return_value = False

        result = orch.process_messages(
            messages=[{"role": "user", "content": "hello"}],
            agent_mode=True,
        )

        assert result.quality_gate_passed is True

    def test_agent_mode_false_performs_normal_checks(self) -> None:
        """When agent_mode=False, normal L4/L5 checks run as usual."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        orch.resource_model = MagicMock()
        orch.resource_model.track_consumption = MagicMock()

        mock_result = MagicMock()
        mock_result.finish_reason = "stop"
        mock_result.tool_calls = None
        mock_result.content = "result = eval('2+2')"
        mock_result.model_used = "minimax"
        mock_result.tokens_used = 20
        orch.executor = MagicMock()
        orch.executor.execute_messages.return_value = mock_result

        orch.scorer = MagicMock()
        orch.scorer.score.return_value = MagicMock(
            total=3, tier="LOW", task_type="general"
        )

        orch.compression_trigger = MagicMock()
        orch.compression_trigger.should_compress.return_value = False

        result = orch.process_messages(
            messages=[{"role": "user", "content": "hello"}],
            agent_mode=False,
        )

        assert result.quality_gate_passed is False

    def test_process_messages_accepts_agent_mode_param(self) -> None:
        """process_messages should accept agent_mode parameter without error."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        orch.resource_model = MagicMock()
        orch.resource_model.track_consumption = MagicMock()

        mock_result = MagicMock()
        mock_result.finish_reason = "stop"
        mock_result.tool_calls = None
        mock_result.content = "ok"
        mock_result.model_used = "minimax"
        mock_result.tokens_used = 5
        orch.executor = MagicMock()
        orch.executor.execute_messages.return_value = mock_result

        orch.scorer = MagicMock()
        orch.scorer.score.return_value = MagicMock(
            total=3, tier="LOW", task_type="general"
        )

        result = orch.process_messages(
            messages=[{"role": "user", "content": "hi"}],
            agent_mode=True,
        )

        assert result.final_output == "ok"


class TestMCPToolGracefulDegradation:
    """Test MCP graceful degradation when mcp package is unavailable."""

    def test_register_mcp_tools_no_mcp_package_logs_warning(self) -> None:
        """register_mcp_tools should not raise when MCP package is absent."""
        import logging
        from unittest.mock import patch

        with patch("importlib.util.find_spec", return_value=None), \
             patch.object(logging.getLogger("blend"), "warning") as mock_warn:
            from blend.core import tool_executor
            tool_executor.register_mcp_tools([])
            mock_warn.assert_called_once()

    def test_mcp_tool_handler_returns_error_when_mcp_unavailable(self) -> None:
        """MCP handler returns error message when MCP package not available."""
        from unittest.mock import patch

        with patch("importlib.util.find_spec", return_value=None):
            from blend.core.tool_executor import _mcp_tool_handler
            handler = _mcp_tool_handler("test_server", "test_tool")
            result = handler({"arg": "value"})
        assert "Error" in result
        assert "not available" in result

    def test_register_mcp_tools_handles_subprocess_failure(self) -> None:
        """register_mcp_tools handles subprocess spawn failure gracefully."""
        import logging
        from unittest.mock import patch

        with patch("importlib.util.find_spec", return_value=True), \
             patch("subprocess.Popen", side_effect=OSError("no such command")), \
             patch.object(logging.getLogger("blend"), "warning") as mock_warn:
            from blend.core import tool_executor
            tool_executor.register_mcp_tools([{"name": "bad", "command": "nonexistent"}])
            assert any("bad" in str(c) for c in mock_warn.call_args_list)


class TestMCPToolRegistration:
    """Test MCP tool registration in tool_executor."""

    def test_register_mcp_tools_function_exists(self) -> None:
        """register_mcp_tools should be importable from tool_executor."""
        from blend.core import tool_executor

        assert hasattr(tool_executor, "register_mcp_tools")
        assert callable(tool_executor.register_mcp_tools)

    def test_register_tool_decorator_adds_to_registry(self) -> None:
        """register_tool decorator should add handler to registry."""
        from blend.core import tool_executor

        call_log: list[dict[str, Any]] = []

        @tool_executor.register_tool("test_echo")  # type: ignore[untyped-decorator]
        def echo_handler(args: dict[str, Any]) -> str:
            call_log.append(args)
            return f"echo: {args.get('value', '')}"

        result = tool_executor.execute_single_tool(
            tool_call={
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_echo", "arguments": {"value": "hello"}},
            },
            tools=None,
        )

        assert result["role"] == "tool"
        assert result["content"] == "echo: hello"
        assert call_log == [{"value": "hello"}]

    def test_execute_tool_calls_runs_in_order(self) -> None:
        """Multiple tool calls should execute in order."""
        from blend.core import tool_executor

        execution_order: list[str] = []

        @tool_executor.register_tool("order_test_a")  # type: ignore[untyped-decorator]
        def handler_a(args: dict[str, Any]) -> str:
            execution_order.append("a")
            return "a"

        @tool_executor.register_tool("order_test_b")  # type: ignore[untyped-decorator]
        def handler_b(args: dict[str, Any]) -> str:
            execution_order.append("b")
            return "b"

        results = tool_executor.execute_tool_calls(
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "order_test_a", "arguments": {}}},
                {"id": "c2", "type": "function", "function": {"name": "order_test_b", "arguments": {}}},
            ],
            tools=None,
        )

        assert execution_order == ["a", "b"]
        assert len(results) == 2
        assert results[0]["content"] == "a"
        assert results[1]["content"] == "b"


class TestContextManager:
    """Test context budget management."""

    def test_estimate_tokens_from_messages(self) -> None:
        """estimate_tokens should return rough token count for messages."""
        from blend.core.context_manager import estimate_tokens_from_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        tokens = estimate_tokens_from_messages(messages)
        assert tokens > 0

    def test_truncate_messages_preserves_recent(self) -> None:
        """truncate_messages should keep the most recent N tokens worth of messages."""
        from blend.core.context_manager import truncate_messages

        messages = [
            {"role": "system", "content": "x" * 800},
            {"role": "user", "content": "y" * 800},
            {"role": "assistant", "content": "z" * 800},
            {"role": "user", "content": "a" * 800},
            {"role": "assistant", "content": "b" * 800},
            {"role": "user", "content": "c" * 800},
        ]

        truncated = truncate_messages(messages, max_tokens=300)

        assert truncated[-1] == messages[-1]
        assert len(truncated) < len(messages)

    def test_truncate_messages_empty_list(self) -> None:
        """truncate_messages should handle empty list gracefully."""
        from blend.core.context_manager import truncate_messages

        result = truncate_messages([], max_tokens=100)
        assert result == []

    def test_truncate_messages_preserves_structure(self) -> None:
        """Truncated messages should retain their dict structure."""
        from blend.core.context_manager import truncate_messages

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello", "tool_calls": None},
            {"role": "assistant", "content": "Hi"},
            {"role": "tool", "tool_call_id": "abc", "content": "result"},
        ]
        truncated = truncate_messages(messages, max_tokens=50)
        for msg in truncated:
            assert "role" in msg
            assert "content" in msg


class TestAgentModeIntegration:
    """End-to-end tests for agent_mode behavior."""

    def test_agent_mode_with_tool_loop_accumulates_messages(self) -> None:
        """In agent mode, tool loop should still accumulate messages correctly."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        orch.resource_model = MagicMock()
        orch.resource_model.track_consumption = MagicMock()

        first_output = MagicMock()
        first_output.finish_reason = "tool_calls"
        first_output.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'},
            }
        ]
        first_output.content = "Let me calculate that."
        first_output.model_used = "minimax"
        first_output.tokens_used = 50

        second_output = MagicMock()
        second_output.finish_reason = "stop"
        second_output.tool_calls = None
        second_output.content = "The answer is 4."
        second_output.model_used = "minimax"
        second_output.tokens_used = 20

        orch.executor = MagicMock()
        orch.executor.execute_messages.side_effect = [first_output, second_output]

        orch.scorer = MagicMock()
        orch.scorer.score.return_value = MagicMock(
            total=3, tier="LOW", task_type="general"
        )

        result = orch.process_messages(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[{"type": "function", "function": {"name": "calculator", "parameters": {}}}],
            agent_mode=True,
        )

        assert orch.executor.execute_messages.call_count == 2
        assert result.tool_call_count == 1
        assert result.tool_loop_iterations == 1
        assert result.final_output == "The answer is 4."

    def test_non_agent_mode_passes_tools_to_executor(self) -> None:
        """agent_mode=False should still pass tools to executor normally."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        orch.resource_model = MagicMock()
        orch.resource_model.track_consumption = MagicMock()

        mock_result = MagicMock()
        mock_result.finish_reason = "stop"
        mock_result.tool_calls = None
        mock_result.content = "result"
        mock_result.model_used = "minimax"
        mock_result.tokens_used = 5
        orch.executor = MagicMock()
        orch.executor.execute_messages.return_value = mock_result

        orch.scorer = MagicMock()
        orch.scorer.score.return_value = MagicMock(
            total=3, tier="LOW", task_type="general"
        )

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        orch.process_messages(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=tools,
            agent_mode=False,
        )

        call_kwargs = orch.executor.execute_messages.call_args.kwargs
        assert call_kwargs["tools"] == tools


class TestCheckContextBudget:
    """Test check_context_budget utility."""

    def test_within_budget_returns_true(self) -> None:
        """check_context_budget returns True when under 80% of context limit."""
        from blend.core.context_manager import check_context_budget

        messages = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
        ]
        assert check_context_budget(messages) is True

    def test_over_budget_returns_false(self) -> None:
        """check_context_budget returns False when over threshold."""
        from blend.core.context_manager import check_context_budget

        messages = [
            {"role": "user", "content": "x" * 500000},
            {"role": "assistant", "content": "y" * 500000},
        ]
        assert check_context_budget(messages) is False

    def test_custom_context_limit_and_percent(self) -> None:
        """check_context_budget respects custom limit and usage percent."""
        from blend.core.context_manager import check_context_budget

        messages = [
            {"role": "user", "content": "x" * 400},
        ]
        assert check_context_budget(messages, context_limit=100, usage_percent=1.0) is True
        assert check_context_budget(messages, context_limit=10, usage_percent=1.0) is False

    def test_empty_messages_within_budget(self) -> None:
        """Empty message list is within budget."""
        from blend.core.context_manager import check_context_budget

        assert check_context_budget([]) is True


class TestCalculatorEvalSafety:
    """Test calculator eval safety boundaries."""

    def test_calc_rejects_unsafe_expression(self) -> None:
        """Calc rejects expressions with non-numeric characters."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_bad",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": "__import__(\'os\')"}'},
        }
        result = execute_single_tool(tool_call)
        assert "Error" in result["content"]

    def test_calc_handles_empty_expression(self) -> None:
        """Calc handles empty expression gracefully."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_empty",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": ""}'},
        }
        result = execute_single_tool(tool_call)
        assert "Error" in result["content"]

    def test_calc_safe_functions_allowed(self) -> None:
        """Calc allows safe numeric expressions (no alphanumeric identifiers)."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_expr",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": "16**0.5"}'},
        }
        result = execute_single_tool(tool_call)
        assert result["content"] == "4.0"

    def test_calc_power_operator_allowed(self) -> None:
        """Calc allows power operator ** (not pow function)."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_pow",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": "2**10"}'},
        }
        result = execute_single_tool(tool_call)
        assert result["content"] == "1024"

    def test_calc_handles_scientific_notation(self) -> None:
        """Calc handles scientific notation."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_sci",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": "1e5 + 1e3"}'},
        }
        result = execute_single_tool(tool_call)
        assert result["content"] == "101000.0"

    def test_calc_rejects_alphanumeric_identifier(self) -> None:
        """Calc rejects identifiers like 'abc' in expression."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_alpha",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": "abc"}'},
        }
        result = execute_single_tool(tool_call)
        assert "Error" in result["content"]


class TestHTTPRequestTool:
    """Test http_request tool execution."""

    def test_http_get_success(self) -> None:
        """http_request GET should return response body."""
        from unittest.mock import patch

        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_http",
            "type": "function",
            "function": {"name": "http_request", "arguments": '{"url": "https://example.com"}'},
        }
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = __import__("io").BytesIO(b'{"ok": true}')
            mock_resp.read = lambda: b'{"ok": true}'
            mock_urlopen.return_value.__enter__ = lambda s: mock_resp
            mock_urlopen.return_value.__exit__ = lambda *a: None
            result = execute_single_tool(tool_call)
        assert "ok" in result["content"]

    def test_http_missing_url(self) -> None:
        """http_request without url should return error."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_http",
            "type": "function",
            "function": {"name": "http_request", "arguments": "{}"},
        }
        result = execute_single_tool(tool_call)
        assert "Error" in result["content"]

    def test_http_unsupported_method(self) -> None:
        """http_request with unsupported method should return error."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_http",
            "type": "function",
            "function": {"name": "http_request", "arguments": '{"url": "https://example.com", "method": "PATCH"}'},
        }
        result = execute_single_tool(tool_call)
        assert "Error" in result["content"]

    def test_http_invalid_json(self) -> None:
        """http_request with invalid JSON should return error."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_http",
            "type": "function",
            "function": {"name": "http_request", "arguments": "not json"},
        }
        result = execute_single_tool(tool_call)
        assert "Error" in result["content"]


class TestMCPToolHandlerWithMCP:
    """Test _mcp_tool_handler when MCP package IS available."""

    def test_mcp_handler_returns_json_when_mcp_available(self) -> None:
        """_mcp_tool_handler returns JSON when MCP package is installed."""
        from unittest.mock import patch

        from blend.core.tool_executor import _mcp_tool_handler

        handler = _mcp_tool_handler("my_server", "my_tool")
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = True
            result = handler({"arg": "val"})
        import json
        parsed = json.loads(result)
        assert parsed["server"] == "my_server"
        assert parsed["tool"] == "my_tool"
        assert parsed["args"] == {"arg": "val"}
