"""Additional tests for tool_executor module to improve coverage."""

from typing import Any
from unittest.mock import MagicMock, patch

from blend.core.tool_executor import (
    _TOOL_REGISTRY,
    ToolError,
    _calc_handler,
    _http_handler,
    _mcp_tool_handler,
    execute_single_tool,
    execute_tool_calls,
    register_tool,
)


class TestCalculatorHandler:
    """Test calculator handler."""

    def test_basic_arithmetic(self) -> None:
        """Basic math works."""
        result = _calc_handler({"expr": "2 + 2"})
        assert result == "4"

    def test_complex_expression(self) -> None:
        """Complex expressions work."""
        result = _calc_handler({"expr": "(10 + 5) * 2 / 3"})
        assert abs(float(result) - 10.0) < 0.01

    def test_sqrt_unsafe_expression(self) -> None:
        """sqrt raises ToolError because sqrt is not in allowed set."""
        try:
            _calc_handler({"expr": "sqrt(16)"})
            assert False, "Should have raised"
        except ToolError as e:
            assert "Unsafe expression" in str(e)

    def test_pi_unsafe_expression(self) -> None:
        """pi raises ToolError because pi is not in allowed set."""
        try:
            _calc_handler({"expr": "pi"})
            assert False, "Should have raised"
        except ToolError as e:
            assert "Unsafe expression" in str(e)

    def test_power(self) -> None:
        """Power function works using ** operator."""
        result = _calc_handler({"expr": "2**3"})
        assert result == "8"

    def test_unsafe_expression_raises(self) -> None:
        """Unsafe expressions raise ToolError."""
        try:
            _calc_handler({"expr": "import os; os.system('rm -rf /')"})
            assert False, "Should have raised"
        except ToolError as e:
            assert "Unsafe expression" in str(e)

    def test_string_argument(self) -> None:
        """Handler accepts string arguments."""
        result = _calc_handler('{"expr": "5 * 5"}')
        assert result == "25"

    def test_division_by_zero(self) -> None:
        """Division by zero raises error."""
        try:
            _calc_handler({"expr": "1 / 0"})
            assert False
        except ToolError as e:
            assert "Calculation error" in str(e)


class TestHttpHandler:
    """Test HTTP request handler."""

    def test_missing_url(self) -> None:
        """Missing URL raises error."""
        try:
            _http_handler({})
            assert False
        except ToolError as e:
            assert "Missing 'url'" in str(e)

    def test_unsupported_method(self) -> None:
        """Unsupported method raises error."""
        try:
            _http_handler({"url": "https://example.com", "method": "PATCH"})
            assert False
        except ToolError as e:
            assert "Unsupported HTTP method" in str(e)

    @patch("urllib.request.urlopen")
    def test_successful_get(self, mock_urlopen: MagicMock) -> None:
        """Successful GET request."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = _http_handler({"url": "https://api.example.com/data"})
        assert "ok" in result

    @patch("urllib.request.urlopen")
    def test_http_error_response(self, mock_urlopen: MagicMock) -> None:
        """HTTP error returns error message."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 404, "Not Found", {}, None
        )

        result = _http_handler({"url": "https://example.com"})
        assert "404" in result

    @patch("urllib.request.urlopen")
    def test_http_request_with_dict_body(self, mock_urlopen: MagicMock) -> None:
        """HTTP request with dict body is converted to JSON."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"ok": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = _http_handler({"url": "https://example.com", "body": {"key": "value"}, "method": "POST"})
        assert isinstance(result, str)


class TestRegisterTool:
    """Test tool registration."""

    def test_register_with_decorator(self) -> None:
        """@register_tool decorator works."""
        @register_tool
        def my_custom_tool(args: dict[str, Any]) -> str:
            return "custom result"

        assert "my_custom_tool" in _TOOL_REGISTRY
        assert _TOOL_REGISTRY["my_custom_tool"]["handler"] == my_custom_tool

    def test_register_with_name(self) -> None:
        """register_tool with explicit name works via decorator."""
        @register_tool("my_named_tool")
        def my_handler(args: dict[str, Any]) -> str:
            return "named"

        assert "my_named_tool" in _TOOL_REGISTRY

    def test_register_with_definition(self) -> None:
        """register_tool with custom definition works."""
        def handler(args: dict[str, Any]) -> str:
            return "defined"

        register_tool(
            "my_defined_tool",
            definition={
                "type": "function",
                "function": {
                    "name": "my_defined_tool",
                    "description": "A custom tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )(handler)

        assert "my_defined_tool" in _TOOL_REGISTRY

    def test_register_decorator_with_parens_no_args(self) -> None:
        """@register_tool() with no args works."""
        @register_tool()
        def another_tool(args: dict[str, Any]) -> str:
            return "result"

        assert "another_tool" in _TOOL_REGISTRY


class TestMcpToolHandler:
    """Test MCP tool handler creation."""

    def test_mcp_handler_returns_error_without_mcp_package(self) -> None:
        """MCP handler returns error when MCP is not installed."""
        with patch("importlib.util.find_spec", return_value=None):
            handler = _mcp_tool_handler("test_server", "test_tool")
            result = handler({"arg1": "value1"})
            # MCP package is not installed, so returns error message
            assert "MCP package not available" in result

    def test_mcp_handler_string_args(self) -> None:
        """MCP handler handles string arguments."""
        with patch("importlib.util.find_spec", return_value=None):
            handler = _mcp_tool_handler("server", "tool")
            result = handler('{"key": "value"}')
            # Without MCP package, returns error message
            assert "MCP package not available" in result


class TestExecuteSingleTool:
    """Additional execute_single_tool tests."""

    def test_dict_arguments(self) -> None:
        """Dict arguments work."""
        tool_call = {
            "id": "call_dict",
            "function": {"name": "calculator", "arguments": {"expr": "10 * 10"}},
        }
        result = execute_single_tool(tool_call)
        assert result["content"] == "100"

    def test_empty_arguments(self) -> None:
        """Empty dict arguments work."""
        tool_call = {
            "id": "call_empty",
            "function": {"name": "calculator", "arguments": {}},
        }
        result = execute_single_tool(tool_call)
        assert result["role"] == "tool"

    def test_tool_error_caught(self) -> None:
        """ToolError is caught and returned as content."""
        # Register a tool that raises ToolError
        def bad_tool(args: Any) -> str:
            raise ToolError("Something went wrong")

        register_tool("bad_tool")(bad_tool)

        tool_call = {
            "id": "call_bad",
            "function": {"name": "bad_tool", "arguments": "{}"},
        }
        result = execute_single_tool(tool_call)
        assert "Error: Something went wrong" in result["content"]

    def test_generic_exception_caught(self) -> None:
        """Generic exceptions are caught."""
        def raising_tool(args: Any) -> str:
            raise ValueError("Unexpected error")

        register_tool("raising_tool")(raising_tool)

        tool_call = {
            "id": "call_raise",
            "function": {"name": "raising_tool", "arguments": "{}"},
        }
        result = execute_single_tool(tool_call)
        assert "ValueError" in result["content"]

    def test_unknown_function_name(self) -> None:
        """Unknown function returns error."""
        tool_call = {
            "id": "call_unknown",
            "function": {"name": "completely_unknown_function", "arguments": "{}"},
        }
        result = execute_single_tool(tool_call)
        assert "Unknown tool" in result["content"]


class TestExecuteToolCalls:
    """Test execute_tool_calls function."""

    def test_empty_list(self) -> None:
        """Empty list returns empty."""
        results = execute_tool_calls([])
        assert results == []

    def test_multiple_tool_calls(self) -> None:
        """Multiple tool calls are executed in order."""
        tool_calls = [
            {"id": "c1", "function": {"name": "calculator", "arguments": {"expr": "1+1"}}},
            {"id": "c2", "function": {"name": "calculator", "arguments": {"expr": "2+2"}}},
            {"id": "c3", "function": {"name": "calculator", "arguments": {"expr": "3+3"}}},
        ]
        results = execute_tool_calls(tool_calls)
        assert len(results) == 3
        assert results[0]["content"] == "2"
        assert results[1]["content"] == "4"
        assert results[2]["content"] == "6"

    def test_mixed_success_and_error(self) -> None:
        """Mixed results work."""
        tool_calls = [
            {"id": "c1", "function": {"name": "calculator", "arguments": {"expr": "5+5"}}},
            {"id": "c2", "function": {"name": "nonexistent", "arguments": {}}},
        ]
        results = execute_tool_calls(tool_calls)
        assert results[0]["content"] == "10"
        assert "Unknown tool" in results[1]["content"]
