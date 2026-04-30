"""Tool Executor - Handles tool call execution in the agentic loop."""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import re
import urllib.request
from typing import Any


class ToolError(Exception):
    """Raised when a tool execution fails."""


# -----------------------------------------------------------------------
# SSRF Protection - Block internal/private IP ranges
# -----------------------------------------------------------------------

_BLOCKED_IP_RANGES = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
)


def _is_blocked_url(url: str) -> str | None:
    """Check if URL resolves to a blocked internal IP.

    Returns None if allowed, or an error string if blocked.
    """
    try:
        parsed = urllib.request.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Only HTTP/HTTPS allowed, got '{parsed.scheme}'"

        host = parsed.hostname
        if not host:
            return "Missing hostname in URL"

        # Resolve hostname to IP(s)
        try:
            addr_info = urllib.request.urlopen(
                urllib.request.Request(f"{parsed.scheme}://{host}", method="HEAD"),
                timeout=3,
            )
            # urlopen doesn't give us the resolved IP easily, use socket
        except Exception as e:
            logger = logging.getLogger("blend")
            logger.debug(f"URL open check failed for {host}: {e}")

        # Use socket to resolve and check
        import socket

        try:
            resolved_ips = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return None  # Can't resolve, let it fail naturally

        for family, _, _, _, sockaddr in resolved_ips:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
                for blocked_net in _BLOCKED_IP_RANGES:
                    if ip in blocked_net:
                        return f"Blocked internal IP range: {ip_str}"
            except ValueError:
                continue

        return None
    except Exception as e:
        logger = logging.getLogger("blend")
        logger.debug(f"IP validation failed for {host}: {e}")
        return None  # On any error, let request proceed (will fail naturally)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(name: str | None = None, *, definition: dict[str, Any] | None = None) -> Any:
    """Register a tool handler.

    Can be used as a decorator with inferred name/definition:
        @register_tool()
        def my_handler(args): ...

    Or with explicit args:
        register_tool("my_tool", definition={...}, handler=fn)

    Args:
        name: Tool name (inferred from handler __name__ if None)
        definition: Tool definition dict (auto-built if None)
        handler: Handler function (the decorated function if using @)

    Returns:
        Decorator identity function or registers immediately
    """
    def decorator(handler: Any) -> Any:
        tool_name = name or handler.__name__
        tool_def = definition or {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Tool: {tool_name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        _TOOL_REGISTRY[tool_name] = {
            "definition": tool_def,
            "handler": handler,
        }
        return handler

    # If called with positional handler (e.g. @register_tool with no parens), call decorator
    if callable(name):
        handler = name
        name = None
        return decorator(handler)

    # If called as @register_tool() with parens but no args: just return decorator
    if name is None and definition is None and not callable(name):
        return decorator

    # Called with explicit args: register immediately
    if name is not None:
        tool_def = definition or {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool: {name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        def named_handler(handler: Any) -> Any:
            _TOOL_REGISTRY[name] = {"definition": tool_def, "handler": handler}
            return handler
        return named_handler

    return decorator


def register_mcp_tools(mcp_servers: list[dict[str, Any]]) -> None:
    """Connect to MCP server(s) and register discovered tools.

    Each server config: {"name": "...", "command": "npx", "args": ["-y", "@server", "/path"]}

    Gracefully degrades: if MCP package not available, logs warning and skips.

    Args:
        mcp_servers: List of MCP server configurations
    """
    import logging

    try:
        import importlib.util
        mcp_spec = importlib.util.find_spec("mcp")
    except Exception:
        mcp_spec = None

    if mcp_spec is None:
        logging.getLogger("blend").warning(
            "MCP package not installed. Install with: pip install mcp. "
            "Skipping MCP tool registration."
        )
        return

    # MCP client implementation using SSE stream protocol
    for server in mcp_servers:
        server_name = server.get("name", "unknown")
        command = server.get("command", "")
        args = server.get("args", [])

        # Attempt connection and tool discovery
        try:
            _discover_mcp_tools(command, args, server_name)
        except Exception:
            logging.getLogger("blend").warning(
                f"Failed to connect to MCP server '{server_name}': {command} {' '.join(args)}"
            )


def _discover_mcp_tools(command: str, args: list[str], server_name: str) -> None:
    """Discover and register tools from an MCP server.

    NOTE: This is a partial implementation. The MCP protocol client
    spawns the server and performs initial handshake (initialize, tools/list)
    but the actual tool call routing (_mcp_tool_handler) is a placeholder
    that echoes arguments rather than making real protocol calls.

    To complete MCP integration:
    1. Replace _mcp_tool_handler to use the MCP client library
       (e.g., from mcp import Client; client.call(tool_name, args))
    2. Maintain a persistent connection per server instead of spawning per discovery
    3. Handle MCP protocol error responses and reconnection
    """
    import json
    import subprocess

    # MCP protocol: spawn server, send initialize → tools/list → disconnect
    # For now, register a placeholder that routes through execute_single_tool
    # The actual MCP protocol handling is deferred to a follow-up integration
    try:
        proc = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.stdout and proc.stdin:
            # Send JSON-RPC initialize
            init = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
            }
            print(json.dumps(init), file=proc.stdin)
            proc.stdin.flush()

            # Read response
            import select
            if select.select([proc.stdout], [], [], 5.0)[0]:
                json.loads(proc.stdout.readline())  # consume initialize response

            # Send tools/list
            list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            print(json.dumps(list_req), file=proc.stdin)
            proc.stdin.flush()

            if select.select([proc.stdout], [], [], 5.0)[0]:
                tools_resp = json.loads(proc.stdout.readline())
                tools = tools_resp.get("result", {}).get("tools", [])
                for tool in tools:
                    tool_name = tool.get("name", "")
                    _TOOL_REGISTRY[f"mcp_{server_name}_{tool_name}"] = {
                        "definition": {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": tool.get("description", f"MCP tool: {tool_name}"),
                                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                            },
                        },
                        "handler": _mcp_tool_handler(server_name, tool_name),
                        "mcp_server": server_name,
                    }
    except Exception:
        pass  # Graceful degradation
    finally:
        if proc:
            proc.terminate()


def _mcp_tool_handler(server_name: str, tool_name: str) -> Any:
    """Create a handler that routes tool calls to an MCP server.

    ALPHA: This is a placeholder implementation. The actual MCP protocol
    call is not yet implemented. Currently returns a JSON echo of args.
    """
    def handler(arguments: dict[str, Any] | str) -> str:
        """Execute an MCP tool via the MCP server."""
        if isinstance(arguments, dict):
            args_dict = arguments
        else:
            args_dict = json.loads(arguments) if isinstance(arguments, str) else {}

        try:
            import importlib.util
            if importlib.util.find_spec("mcp") is None:
                return f"Error: MCP package not available for {server_name}"
            # MCP package is available — placeholder for full MCP client integration
            # In a real implementation, would create: Client(server_name, tool_name).call(args_dict)
            return json.dumps({"server": server_name, "tool": tool_name, "args": args_dict})
        except Exception:
            return f"Error: Failed to call MCP tool {tool_name}"

    return handler


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _calc_handler(arguments: dict[str, Any] | str) -> str:
    """Evaluate a safe subset of math expressions using AST parsing (no eval)."""
    import ast
    import operator

    if isinstance(arguments, dict):
        expr = str(arguments.get("expr", ""))
    else:
        # Legacy: JSON string
        parsed = json.loads(arguments)
        expr = str(parsed.get("expr", ""))

    # Only allow safe math operations
    allowed = re.compile(r"^[0-9+\-*/().eE\s]+$")
    if not allowed.match(expr):
        raise ToolError(f"Unsafe expression: {expr}")

    try:
        # Parse into AST - this catches syntactic errors before we evaluate
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"Syntax error: {e}")

    # Define safe operations
    safe_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Constants
    safe_consts = {
        "math.pi": math.pi,
        "math.e": math.e,
    }

    def eval_node(node: ast.AST) -> float | int:
        """Recursively evaluate AST node with only safe operations."""
        match node:
            case ast.Constant(value=value) if isinstance(value, int):
                return value  # Preserve int
            case ast.Constant(value=value) if isinstance(value, float):
                return value
            case ast.BinOp(left=left, op=op, right=right) if type(op) in safe_ops:
                lv = eval_node(left)
                rv = eval_node(right)
                try:
                    result = safe_ops[type(op)](lv, rv)
                except ZeroDivisionError:
                    raise ToolError("Calculation error: division by zero")
                # Preserve int for power of ints with whole result
                if type(op) is ast.Pow and isinstance(lv, int) and isinstance(rv, int):
                    return int(result)
                return result
            case ast.UnaryOp(op=op, operand=operand) if type(op) in safe_ops:
                return safe_ops[type(op)](eval_node(operand))
            case ast.Call(func=func, args=[arg]) if (
                isinstance(func, ast.Attribute) and
                func.attr in ("sqrt",) and
                isinstance(func.value, ast.Name) and
                func.value.id == "math"
            ):
                return getattr(math, func.attr)(eval_node(arg))
            case _:
                raise ToolError(f"Unsupported expression node: {ast.dump(node)}")

    result = eval_node(tree.body)
    return str(result)


def _http_handler(arguments: dict[str, Any] | str) -> str:
    """Make an HTTP request (GET or POST) with SSRF protection."""
    try:
        if isinstance(arguments, dict):
            data = arguments
        else:
            data = json.loads(arguments)
    except json.JSONDecodeError:
        raise ToolError("Invalid JSON for http_request")

    url = data.get("url")
    method = data.get("method", "GET").upper()
    headers = data.get("headers", {})
    body = data.get("body")

    if not url:
        raise ToolError("Missing 'url' in http_request")

    if method not in ("GET", "POST", "PUT", "DELETE"):
        raise ToolError(f"Unsupported HTTP method: {method}")

    # SSRF protection: block internal/private IPs
    blocked_reason = _is_blocked_url(url)
    if blocked_reason:
        raise ToolError(f"SSRF blocked: {blocked_reason}")

    req = urllib.request.Request(url, method=method)
    for k, v in headers.items():
        req.add_header(k, v)

    if body and method in ("POST", "PUT"):
        body_bytes = json.dumps(body).encode() if isinstance(body, dict) else str(body).encode()
        req.add_header("Content-Type", "application/json")
        req.data = body_bytes

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_body: str = resp.read().decode("utf-8", errors="replace")
            return resp_body[:2000]  # Cap at 2000 chars
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        raise ToolError(f"HTTP request failed: {e}")


# Register built-ins using the register_tool decorator
register_tool(
    "calculator",
    definition={
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "Math expression, e.g. '2+2'"}},
                "required": ["expr"],
            },
        },
    },
)(_calc_handler)


register_tool(
    "http_request",
    definition={
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make an HTTP GET/POST/PUT/DELETE request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                    "headers": {"type": "object"},
                    "body": {"type": "object"},
                },
                "required": ["url"],
            },
        },
    },
)(_http_handler)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_single_tool(
    tool_call: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a single tool call.

    Args:
        tool_call: dict with id, type, function{name, arguments}
        tools: available tool definitions (for validation)

    Returns:
        Tool result message dict: {role: "tool", tool_call_id, content}
    """
    call_id = tool_call.get("id", "unknown")
    func = tool_call.get("function", {})
    func_name = func.get("name", "unknown")
    raw_args = func.get("arguments", "{}")

    # Parse arguments
    try:
        if isinstance(raw_args, str):
            arguments = json.loads(raw_args)
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {}
    except json.JSONDecodeError:
        return _error_result(call_id, f"Error: Invalid arguments JSON: {raw_args[:100]}")

    # Look up handler in registry
    handler_info = _TOOL_REGISTRY.get(func_name)
    if handler_info is None:
        # Try to match by name in provided tools list
        if tools:
            for t in tools:
                t_func = t.get("function", {})
                if t_func.get("name") == func_name:
                    # No handler registered — return a placeholder explaining the limitation
                    return _error_result(
                        call_id,
                        f"Error: Tool '{func_name}' is defined but not registered for local execution. "
                        f"Arguments received: {json.dumps(arguments)}",
                    )
        return _error_result(call_id, f"Error: Unknown tool '{func_name}'")

    # Execute handler
    try:
        handler = handler_info["handler"]
        if callable(handler):
            # Built-in registered handler: pass parsed dict
            result = handler(arguments)
        else:
            # String-based handler (legacy): pass JSON string
            result = handler(json.dumps(arguments))
        content = str(result)
    except ToolError as e:
        content = f"Error: {e}"
    except Exception as e:
        content = f"Error: {type(e).__name__}: {e}"

    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }


def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Execute all tool calls and return tool result messages.

    Args:
        tool_calls: list of tool call dicts from model response
        tools: available tool definitions

    Returns:
        list of tool result message dicts (in order)
    """
    results: list[dict[str, Any]] = []
    for tc in tool_calls:
        results.append(execute_single_tool(tc, tools))
    return results


def _error_result(tool_call_id: str, content: str) -> dict[str, Any]:
    """Create an error tool result."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
