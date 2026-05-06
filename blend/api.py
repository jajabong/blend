"""FastAPI application for blend API - OpenAI compatible."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()  # noqa: E402

from blend.config import require_keys  # noqa: E402
from blend.core.budget import ResourceModel  # noqa: E402
from blend.core.orchestrator import BlendOrchestrator  # noqa: E402

# Validate required configuration on startup
try:
    require_keys("MINIMAX_API_KEY", "BAOSI_API_KEY", "LEMON_API_KEY")
except ValueError as e:
    import warnings

    warnings.warn(f"Configuration warning: {e}")


async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Register MCP tools at startup, not per-request."""
    import logging
    import os
    logger = logging.getLogger("blend")

    # Force clear any cached config
    from blend.config import get_mcp_servers
    get_mcp_servers.cache_clear()

    from blend.core.tool_executor import _TOOL_REGISTRY, register_mcp_tools

    mcp_servers = get_mcp_servers()
    logger.info(f"MCP servers at startup: {mcp_servers}")
    logger.info(f"BLEND_MCP_SERVERS env: {os.environ.get('BLEND_MCP_SERVERS', 'NOT SET')}")
    if mcp_servers:
        register_mcp_tools(mcp_servers)
        logger.info(f"Tools registered at startup: {list(_TOOL_REGISTRY.keys())}")
    else:
        logger.warning("No MCP servers configured")
    yield
    # Cleanup on shutdown if needed


app = FastAPI(
    title="Blend API",
    description="极致成本效率商用 API - 自动智能路由 & 瞬时自愈心脏",
    version="2.1.0",
    lifespan=lifespan,  # type: ignore[arg-type]
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log validation errors for debugging."""
    import logging
    logger = logging.getLogger("blend")
    logger.error(f"Validation error: {exc.errors()}")
    body = await request.body()
    logger.error(f"Request body: {body!r}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode()}
    )

# Global instances
resource_model = ResourceModel()
orchestrator = BlendOrchestrator()


class Message(BaseModel):
    """Chat message."""

    role: str
    content: str | list[dict[str, Any]] | None = None  # Optional for tool calls messages
    tool_call_id: str | None = None
    name: str | None = None


# ─── Anthropic Messages API Models ─────────────────────────────────────────────


class ContentBlockText(BaseModel):
    """Anthropic text content block."""

    type: str = "text"
    text: str


class ContentBlockToolUse(BaseModel):
    """Anthropic tool_use content block."""

    type: str = "tool_use"
    name: str
    input_json: str  # JSON string of tool arguments


class ToolParam(BaseModel):
    """Tool parameter schema."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class AnthropicMessageRequest(BaseModel):
    """Anthropic Messages API request — Claude Code compatible."""

    model: str
    max_tokens: int | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None


class AnthropicUsage(BaseModel):
    """Anthropic usage summary."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicMessageResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[dict[str, Any]]
    model: str
    usage: dict[str, int]
    stop_reason: str | None = None
    stop_sequence: str | None = None


class AnthropicStreamingEvent(BaseModel):
    """Anthropic SSE event wrapper."""

    type: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        """Format as SSE data line."""
        import json
        return f"data: {json.dumps(self.data)}\n\n"


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "blend"  # blend auto-routes
    messages: list[Message]
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None  # OpenAI reasoning models
    reasoning_effort: str | None = None  # OpenAI reasoning effort (low/medium/high)
    temperature: float = 1.0
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    agent_mode: bool = False
    mcp_servers: list[dict[str, Any]] | None = None
    stream_options: dict[str, Any] | None = None  # OpenAI stream options


def process_through_layers(prompt: str) -> tuple[str, dict[str, Any]]:
    """Process prompt through blend 4-layer pipeline.

    Args:
        prompt: User's prompt

    Returns:
        Tuple of (response_text, metadata)
    """
    result = orchestrator.process(prompt)

    return result.final_output, {
        "complexity": result.complexity,
        "layer_path": result.layer_path,
        "model_used": result.model_used,
        "tokens_used": result.tokens_used,
        "quality_gate_passed": result.quality_gate_passed,
        "l1_compressed": result.l1_compressed,
        "l4_applied": result.l4_applied,
    }


def stream_through_layers(prompt: str) -> Generator[str, None, None]:
    """Stream prompt through blend pipeline with real provider streaming.

    Yields SSE-formatted data strings.
    """
    import json
    import time

    chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
    for chunk in orchestrator.stream(prompt):
        choices = chunk.get("choices", [])
        if not choices and "delta" in chunk:
            choices = [{"index": 0, "delta": chunk.get("delta", {}), "finish_reason": chunk.get("finish_reason", "stop")}]
        payload = {
            "id": chunk.get("id", chunk_id),
            "choices": choices,
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health() -> JSONResponse:
    """Health check — reports Circuit Breaker states from the registry."""
    from blend.core.circuit_breaker import get_registry
    registry = get_registry()

    status: dict[str, Any] = {
        "status": "healthy",
        "service": "blend",
        "circuit_breakers": {
            name: {
                "state": b.state.value,
                "consecutive_trips": b._consecutive_trips,
                "lockout": f"{b._lockout_duration}s"
            } for name, b in registry._breakers.items()
        },
    }

    # If all major providers are open, mark as degraded
    if any(b.state.value == "open" for b in registry._breakers.values()):
        status["status"] = "degraded"

    return JSONResponse(status)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    from blend import __version__
    return {"service": "blend", "version": __version__}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest) -> JSONResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    blend auto-routes to optimal model based on task complexity.
    Supports: tools, tool_choice, response_format (JSON mode), multimodal content.
    """
    import logging
    logger = logging.getLogger("blend")
    logger.debug(f"chat_completions request: stream={request.stream}, tools={bool(request.tools)}, tool_choice={request.tool_choice}")
    if request.tools:
        logger.debug(f"tools: {request.tools[:1]}...")  # First tool only
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    # Build message list preserving structure (multimodal, tool results, etc.)
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        msg_dict: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            msg_dict["content"] = m.content
        if m.tool_call_id:
            msg_dict["tool_call_id"] = m.tool_call_id
        if m.name:
            msg_dict["name"] = m.name
        messages.append(msg_dict)

    if request.stream:
        return StreamingResponse(
            _stream_async(
                messages,
                request.tools,
                request.tool_choice,
                request.response_format,
                request.agent_mode,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.presence_penalty,
                request.frequency_penalty,
                request.stop,
            ),
            media_type="text/event-stream",
        )

    try:
        result = await asyncio.to_thread(
            orchestrator.process_messages,
            messages=messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            agent_mode=request.agent_mode,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            stop=request.stop,
        )
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        if status_code == 401 or status_code == 403:
            raise HTTPException(
                status_code=401,
                detail="Authentication failed with upstream provider.",
            )
        if status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": e.response.headers.get("retry-after", "30")},
            )
        if status_code >= 500:
            raise HTTPException(
                status_code=503,
                detail=f"Upstream provider error: {e}",
                headers={"Retry-After": "30"},
            )
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Upstream request timed out. Please try again.",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to upstream provider.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    prompt_tokens = sum(
        len(str(m.get("content", ""))) // 4 if isinstance(m.get("content"), str) else 20
        for m in messages
    )
    completion_tokens = len(result.final_output) // 4

    # Build assistant message
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.final_output}
    if result.tool_calls:
        assistant_msg["tool_calls"] = result.tool_calls

    response_id = f"chatcmpl-{int(time.time() * 1000)}"

    return JSONResponse(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "blend",
            "choices": [
                {
                    "index": 0,
                    "message": assistant_msg,
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "_blend_metadata": {
                "complexity": result.complexity,
                "layer_path": result.layer_path,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "quality_gate_passed": result.quality_gate_passed,
                "l1_compressed": result.l1_compressed,
                "l4_applied": result.l4_applied,
                "tool_call_count": result.tool_call_count,
                "tool_loop_iterations": result.tool_loop_iterations,
                "thought": result.thought,
            },
        }
    )


def _stream_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    response_format: dict[str, Any] | None,
    agent_mode: bool = False,
    max_tokens: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    stop: str | list[str] | None = None,
) -> Generator[str, None, None]:
    """Stream message list through blend pipeline, yielding SSE-formatted strings."""
    import json

    chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
    for chunk in orchestrator.stream_messages(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        agent_mode=agent_mode,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop=stop,
    ):
        choices = chunk.get("choices", [])
        if not choices and "delta" in chunk:
            choices = [{"index": 0, "delta": chunk.get("delta", {}), "finish_reason": chunk.get("finish_reason", "stop")}]
        import time
        payload = {
            "id": chunk.get("id", chunk_id),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": chunk.get("model", "blend"),
            "choices": choices,
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


import os

# SSE heartbeat interval (seconds) - 0 disables heartbeat
SSE_HEARTBEAT_INTERVAL = float(os.environ.get("SSE_HEARTBEAT_INTERVAL", "15"))

# Maximum time between chunks before sending heartbeat
SSE_CHUNK_TIMEOUT = float(os.environ.get("SSE_CHUNK_TIMEOUT", "10"))


def _normalize_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Normalize tool schema for compatibility with all providers.

    Some providers (minimax) reject tools with empty objects {} in properties.
    This function removes empty property definitions and ensures consistent structure.
    """
    if not isinstance(tool, dict):
        return tool

    tool = dict(tool)  # Don't mutate original
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool

    parameters = function.get("parameters", {})
    if not isinstance(parameters, dict):
        return tool

    # Remove properties with empty definitions {}
    properties = parameters.get("properties", {})
    if isinstance(properties, dict):
        cleaned_properties = {
            k: v for k, v in properties.items()
            if v and isinstance(v, dict) and v
        }
        parameters["properties"] = cleaned_properties

    function["parameters"] = parameters
    tool["function"] = function
    return tool


def _normalize_tools_list(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Normalize a list of tools for provider compatibility."""
    if not tools:
        return None
    return [_normalize_tool_schema(t) for t in tools]


def _format_sse_payload(payload: dict[str, Any]) -> str:
    """Format a payload as standard SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _format_sse_comment(message: str) -> str:
    """Format a comment as SSE comment line (starts with :)."""
    return f": {message}\n\n"


def _is_valid_sse_line(line: str) -> bool:
    """Check if a line is valid SSE format."""
    return line.startswith("data: ") or line.startswith(":")


async def _stream_async(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    response_format: dict[str, Any] | None,
    agent_mode: bool = False,
    max_tokens: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    stop: str | list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Async wrapper — runs sync stream in a thread to avoid blocking the event loop.

    Uses an asyncio.Queue to yield chunks in real-time as they arrive from the
    sync generator, rather than buffering the entire stream before sending the
    first byte. This prevents "Connection reset by server" when the tool execution
    loop takes time before producing output.

    Features:
    - Periodic heartbeat comments to keep connection alive
    - Standard SSE format validation
    - Graceful error handling in SSE format
    """
    import json as json_mod

    queue: asyncio.Queue[tuple[bool, str | BaseException]] = asyncio.Queue()
    chunk_id_str = f"chatcmpl-{int(time.time() * 1000)}"
    exc_info: BaseException | None = None
    last_chunk_time = time.monotonic()
    chunk_count = 0

    def sync_producer() -> None:
        """Runs in a thread pool — produces chunks into the async queue."""
        nonlocal exc_info, last_chunk_time, chunk_count
        # Normalize tools before passing to provider (removes empty property objects)
        normalized_tools = _normalize_tools_list(tools)
        try:
            for chunk in orchestrator.stream_messages(
                messages=messages,
                tools=normalized_tools,
                tool_choice=tool_choice,
                response_format=response_format,
                agent_mode=agent_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop,
            ):
                choices = chunk.get("choices", [])
                if not choices and "delta" in chunk:
                    choices = [{"index": 0, "delta": chunk.get("delta", {}), "finish_reason": chunk.get("finish_reason", "stop")}]
                payload = {
                    "id": chunk.get("id", chunk_id_str),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": chunk.get("model", "blend"),
                    "choices": choices,
                }
                last_chunk_time = time.monotonic()
                chunk_count += 1
                sse_line = f"data: {json_mod.dumps(payload)}\n\n"
                # Validate SSE format before putting in queue
                for line in sse_line.strip().split("\n"):
                    if line and not _is_valid_sse_line(line):
                        # Force standard format
                        sse_line = f"data: {json_mod.dumps(payload)}\n\n"
                        break
                queue.put_nowait((False, sse_line))
            queue.put_nowait((False, "data: [DONE]\n\n"))
        except BaseException as e:
            exc_info = e
            queue.put_nowait((True, e))

    # Schedule sync producer in thread pool WITHOUT awaiting it
    # run_in_executor returns a Future immediately; the producer runs concurrently
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, sync_producer)

    # Consume chunks in real-time as they arrive
    while True:
        if exc_info is not None:
            # Circuit breaker or connection error - yield graceful error instead of abrupt close
            err_msg = str(exc_info)
            if "Circuit breaker" in err_msg or "ConnectError" in type(exc_info).__name__:
                error_payload = {
                    "id": chunk_id_str,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"[Service temporarily unavailable: {err_msg}]"},
                        "finish_reason": "error"
                    }],
                    "error": {
                        "message": err_msg,
                        "type": "circuit_breaker_open",
                        "code": 503
                    }
                }
                yield _format_sse_payload(error_payload)
            yield "data: [DONE]\n\n"
            break
        try:
            # Use shorter timeout to allow heartbeat checks
            timeout = min(SSE_CHUNK_TIMEOUT, 30.0)  # Cap at 30s
            is_exc, value = await asyncio.wait_for(queue.get(), timeout=timeout)
            last_chunk_time = time.monotonic()
        except TimeoutError:
            # Send heartbeat if interval elapsed and no chunks received
            if SSE_HEARTBEAT_INTERVAL > 0:
                elapsed = time.monotonic() - last_chunk_time
                if elapsed >= SSE_HEARTBEAT_INTERVAL:
                    yield _format_sse_comment(f"heartbeat {int(elapsed)}s since last chunk")
                    last_chunk_time = time.monotonic()
            continue
        if is_exc:
            # Other error - try graceful message in SSE format
            error_payload = {
                "id": chunk_id_str,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"[Error: {str(value)}]"},
                    "finish_reason": "error"
                }]
            }
            yield _format_sse_payload(error_payload)
            yield "data: [DONE]\n\n"
            break
        yield value  # type: ignore[misc]
        if value == "data: [DONE]\n\n":
            break


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """伪装成标准的 Anthropic 模型列表，让 Claude Code 认为连接的是正版 API。"""
    return {
        "object": "list",
        "data": [
            {
                "id": "claude-haiku-4-5-20251001",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude Haiku 4.5",
            },
            {
                "id": "claude-sonnet-4-6",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude Sonnet 4.6",
            },
            {
                "id": "claude-sonnet-4-6-20250514",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude Sonnet 4.6",
            },
            {
                "id": "claude-opus-4-6",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude Opus 4.6",
            },
            {
                "id": "claude-opus-4-6-20250514",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude Opus 4.6",
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude 3.5 Sonnet",
            },
            {
                "id": "claude-3-5-sonnet-latest",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude 3.5 Sonnet Latest",
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude 3.5 Haiku",
            },
            {
                "id": "claude-3-haiku-20240229",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude 3 Haiku",
            },
            {
                "id": "claude-3-sonnet-20240229",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude 3 Sonnet",
            },
            {
                "id": "claude-3-opus-20240229",
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "description": "Claude 3 Opus",
            }
        ],
    }


@app.get("/v1/budget")
async def get_budget() -> dict[str, Any]:
    """Get current budget status for all models."""
    return {
        "minimax": resource_model.get_status("minimax"),
        "haiku": resource_model.get_status("haiku"),
        "sonnet": resource_model.get_status("sonnet"),
        "opus": resource_model.get_status("opus"),
    }


@app.get("/v1/info")
async def get_info() -> dict[str, Any]:
    """Get blend system information."""
    from blend import __version__
    return {
        "service": "blend",
        "version": __version__,
        "description": "极致成本效率商用 API",
        "layer_architecture": "L1(压缩+评分) > L2(策略,仅HIGH) > L3(执行) > L5(终审)",
        "routing": "Automatic based on complexity and task type",
        "providers": {
            "minimax": "L1 compression + LOW complexity tasks",
            "baosi": "Claude models (Haiku/Sonnet/Opus)",
            "lemon": "Gemini models for deep reasoning",
        },
    }


# ─── Anthropic Messages API Endpoint ─────────────────────────────────────────────


def _build_messages_from_anthropic(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert Anthropic message format to blend internal format.

    - Prepends system prompt as a system role message if provided.
    - Preserves tool_result content blocks from Claude Code.
    """
    result: list[dict[str, Any]] = []

    if system:
        if isinstance(system, list):
            # Join multiple text blocks if it's a list
            system_text = ""
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n"
                elif isinstance(block, str):
                    system_text += block + "\n"
            result.append({"role": "system", "content": system_text.strip()})
        else:
            result.append({"role": "system", "content": system})

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        # Anthropic tool_result: {"role": "user", "content": [{"type": "tool_result", ...}]}
        # Forward as-is (tool_executor handles tool result injection)
        if isinstance(content, list):
            result.append({"role": role, "content": content})
        else:
            result.append({"role": role, "content": content})

    return result


def _convert_chunk_to_anthropic_events(
    chunk: dict[str, Any], is_first: bool = True
) -> Generator[str, None, None]:
    """Convert an OpenAI-style chunk to Anthropic SSE event types.

    Yields formatted SSE data lines (without the 'data: ' prefix — caller adds it).
    """
    import json

    chunk_id = chunk.get("id", "msg_anthropic")
    # Support both formats:
    # 1. Top-level keys (orchestrator.stream_messages): {"delta": {}, "finish_reason": ...}
    # 2. OpenAI choices format: {"choices": [{"delta": {}, "finish_reason": ...}]}
    if "choices" in chunk:
        choices = chunk.get("choices", [])
        choice = choices[0] if choices else {}
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
    else:
        delta = chunk.get("delta", {})
        finish_reason = chunk.get("finish_reason")

    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    mapped_reason = mapping.get(str(finish_reason) if finish_reason else "", "end_turn")

    if is_first:
        # 1. message_start — metadata
        yield json.dumps({
            "type": "message_start",
            "message": {
                "id": chunk_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "blend",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        # 2. content_block_start — text block
        yield json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })

    # 3. content_block_delta — text or tool_use delta
    if "content" in delta and delta["content"]:
        yield json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": delta["content"]},
        })

    # 4. Tool call handling — convert OpenAI tool_calls to Anthropic tool_use
    if "tool_calls" in delta and delta["tool_calls"]:
        for idx, tc in enumerate(delta["tool_calls"]):
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            yield json.dumps({
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "name": func.get("name", ""),
                    "id": tc.get("id", f"toolu_{idx}"),
                },
            })
            yield json.dumps({
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "tool_use_input_json_delta", "input_json": args},
            })

    if finish_reason and finish_reason != "null":
        # content_block_stop
        yield json.dumps({"type": "content_block_stop", "index": 0})

        # message_delta — final usage + stop_reason
        yield json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": mapped_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })

        # message_stop
        yield json.dumps({"type": "message_stop"})


@app.post("/v1/messages", response_model=None)
async def anthropic_messages(
    request: AnthropicMessageRequest,
) -> JSONResponse | StreamingResponse:
    """Anthropic Messages API — Claude Code compatible.

    Accepts the same request schema as the Anthropic Messages API.
    Translates to blend's orchestrator, returns either:
    - JSON message response (stream=False)
    - SSE stream with Anthropic event types (stream=True)
    """
    # Build message list (prepend system prompt as system message)
    messages = _build_messages_from_anthropic(request.messages, request.system)

    if request.stream:
        # Strip tools for streaming since no providers support streaming + tools properly.
        # Tools will be ignored and the model will respond without tool support.
        # For tool execution, use non-streaming mode instead.
        return StreamingResponse(
            _stream_anthropic(messages, None, request.max_tokens,
                              request.temperature, request.top_p,
                              request.stop_sequences),
            media_type="text/event-stream",
        )

    result = orchestrator.process_messages(
        messages=messages,
        tools=request.tools,
        max_tokens=request.max_tokens,
        temperature=request.temperature or 1.0,
        top_p=request.top_p,
        stop=request.stop_sequences,
    )

    # Map blend finish_reason to Anthropic stop_reason
    stop_reason = _map_finish_reason(result.finish_reason)

    response_id = f"msg_{int(time.time() * 1000)}"
    prompt_tokens = sum(
        len(str(m.get("content", ""))) // 4 if isinstance(m.get("content"), str) else 20
        for m in messages
    )
    completion_tokens = len(result.final_output) // 4

    # Build content blocks
    content_blocks: list[dict[str, Any]] = []
    if result.final_output:
        content_blocks.append({"type": "text", "text": result.final_output})

    # Convert OpenAI tool_calls to Anthropic tool_use blocks
    if result.tool_calls:
        for tc in result.tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            content_blocks.append({
                "type": "tool_use",
                "name": func.get("name", ""),
                "input": json.loads(args) if isinstance(args, str) else args,
                "id": tc.get("id", ""),
            })

    return JSONResponse({
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": result.model_used or "blend",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
        "_blend": {
            "complexity": result.complexity,
            "layer_path": result.layer_path,
            "tokens_used": result.tokens_used,
            "quality_gate_passed": result.quality_gate_passed,
            "thought": result.thought,
        },
    })


def _map_finish_reason(finish_reason: str) -> str:
    """Map blend/OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    return mapping.get(finish_reason, "end_turn")


def _stream_result_as_sse(result, tools: list[dict[str, Any]] | None) -> Generator[str, None, None]:
    """Convert a non-streaming orchestrator result to streaming SSE format.

    This is used when tools are requested in a streaming request - we execute
    non-streaming internally (which supports tools properly) and then convert
    the response to streaming SSE.
    """
    import json
    from typing import Any

    response_id = f"msg_{int(__import__('time').time() * 1000)}"

    # 1. message_start
    yield f"data: {json.dumps({
        'type': 'message_start',
        'message': {
            'id': response_id,
            'type': 'message',
            'role': 'assistant',
            'content': [],
            'model': result.model_used or 'blend',
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {'input_tokens': 0, 'output_tokens': 0},
        },
    })}\n\n"

    # 2. content_block_start - check if we have tool_calls
    if result.tool_calls:
        # Stream tool_use events
        for idx, tc in enumerate(result.tool_calls):
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            if isinstance(args_str, str):
                args = json.loads(args_str)
            else:
                args = args_str

            yield f"data: {json.dumps({
                'type': 'content_block_start',
                'index': idx,
                'content_block': {
                    'type': 'tool_use',
                    'name': name,
                    'id': tc.get('id', f'toolu_{idx}'),
                },
            })}\n\n"

            yield f"data: {json.dumps({
                'type': 'content_block_delta',
                'index': idx,
                'delta': {'type': 'tool_use_input_json_delta', 'input_json': json.dumps(args)},
            })}\n\n"

        # content_block_stop
        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

        # message_delta
        yield f"data: {json.dumps({
            'type': 'message_delta',
            'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
            'usage': {'output_tokens': 0},
        })}\n\n"

        # message_stop
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
    else:
        # No tool_calls - stream as text
        yield f"data: {json.dumps({
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'text', 'text': ''},
        })}\n\n"

        # Stream the text content in chunks
        text = result.final_output or ""
        chunk_size = 50  # characters per chunk
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            yield f"data: {json.dumps({
                'type': 'content_block_delta',
                'index': 0,
                'delta': {'type': 'text_delta', 'text': chunk},
            })}\n\n"

        # content_block_stop
        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

        # message_delta
        yield f"data: {json.dumps({
            'type': 'message_delta',
            'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
            'usage': {'output_tokens': 0},
        })}\n\n"

        # message_stop
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"

    yield "data: [DONE]\n\n"


def _parse_tool_call(text: str) -> tuple[str, dict[str, str]] | None:
    """Parse tool call from text format.

    Supports two formats:
    1. [TOOL_CALL]{tool => "name", args => {\n  --key "value"\n}}[/TOOL_CALL]
    2. <invoke="name">\n  --key "value"\n</invoke>

    Returns (tool_name, args_dict) or None if parsing fails.
    """
    import re

    # Format 1: [TOOL_CALL]{tool => "bash", args => {\n  --command "echo hello"\n}}[/TOOL_CALL]
    if "[TOOL_CALL]" in text:
        name_match = re.search(r'tool\s*=>\s*"([^"]+)"', text)
        if name_match:
            tool_name = name_match.group(1)
            args = {}
            arg_matches = re.findall(r'--(\w+)\s+"([^"]*)"', text)
            for key, value in arg_matches:
                args[key] = value
            return tool_name, args
        return None

    # Format 2: <invoke="bash">\n  --command "echo hello"\n</invoke>
    name_match = re.search(r'<invoke="([^"]+)"', text)
    if name_match:
        tool_name = name_match.group(1)
        args = {}
        # Extract --key "value" patterns
        arg_matches = re.findall(r'--(\w+)\s+"([^"]*)"', text)
        for key, value in arg_matches:
            args[key] = value
        return tool_name, args

    return None


def _stream_anthropic(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    stop_sequences: list[str] | None,
) -> Generator[str, None, None]:
    """Stream blend output as Anthropic SSE events.

    Features:
    - Standard SSE format for SDK compatibility
    - Detects [TOOL_CALL]...[/TOOL_CALL] and <invoke>... text formats and converts to tool_use events
    - Minimal heartbeat for very long streams (>50 chunks)
    """
    import re
    is_first = True
    chunk_count = 0
    tool_call_buffer = ""  # Buffer to accumulate text for tool call parsing
    in_tool_call = False   # True when we're inside a tool call block
    tool_call_start_pattern = ""  # The start pattern detected ([TOOL_CALL] or <invoke="...">)
    tool_call_end_pattern = ""   # The end pattern detected ([/TOOL_CALL] or </invoke>)
    tool_call_index = 0    # Index for multiple tool calls
    pending_text_before_tool = ""  # Text before tool call that should be yielded first

    for chunk in orchestrator.stream_messages(
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature or 1.0,
        top_p=top_p,
        stop=stop_sequences,
    ):
        chunk_count += 1

        # Check if this chunk has text content that might contain tool calls
        delta = chunk.get("delta", {})
        content = delta.get("content", "")

        if content:
            tool_call_buffer += content

            # Check for tool call start patterns
            if not in_tool_call:
                tool_call_start = -1
                tool_call_end = -1
                start_pattern = ""
                end_pattern = ""

                # Detect format 1: [TOOL_CALL]...[/TOOL_CALL]
                if "[TOOL_CALL]" in tool_call_buffer:
                    tool_call_start = tool_call_buffer.find("[TOOL_CALL]")
                    start_pattern = "[TOOL_CALL]"
                    end_pattern = "[/TOOL_CALL]"

                # Detect format 2: <invoke="...">...</invoke>
                invoke_match = re.search(r'<invoke="([^"]+)"', tool_call_buffer)
                if invoke_match and (tool_call_start == -1 or invoke_match.start() < tool_call_start):
                    tool_call_start = invoke_match.start()
                    start_pattern = f'<invoke="{invoke_match.group(1)}"'
                    end_pattern = "</invoke>"

                if tool_call_start != -1:
                    end_idx = tool_call_buffer.find(end_pattern, tool_call_start)

                    # Yield any text before the tool call first
                    if tool_call_start > 0:
                        pending_text_before_tool += tool_call_buffer[:tool_call_start]

                    if end_idx != -1:
                        # Complete tool call block
                        tool_call_text = tool_call_buffer[tool_call_start + len(start_pattern):end_idx]
                        tool_call_buffer = tool_call_buffer[end_idx + len(end_pattern):]

                        # Parse and emit tool call
                        parsed = _parse_tool_call(start_pattern + tool_call_text + end_pattern)
                        if parsed:
                            tool_name, args = parsed

                            # Emit any pending text before the tool call
                            if pending_text_before_tool.strip():
                                yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': pending_text_before_tool}})}\n\n"
                                pending_text_before_tool = ""

                            yield f"data: {json.dumps({'type': 'content_block_start', 'index': tool_call_index, 'content_block': {'type': 'tool_use', 'name': tool_name, 'id': f'toolu_{tool_call_index}'}})}\n\n"
                            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': tool_call_index, 'delta': {'type': 'tool_use_input_json_delta', 'input_json': json.dumps(args)}})}\n\n"
                            yield f"data: {json.dumps({'type': 'content_block_stop', 'index': tool_call_index})}\n\n"
                            yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
                            tool_call_index += 1

                        # Process any remaining buffer
                        if tool_call_buffer:
                            pending_text_before_tool += tool_call_buffer
                            tool_call_buffer = ""
                        continue
                    else:
                        # Incomplete tool call - wait for more
                        in_tool_call = True
                        tool_call_start_pattern = start_pattern
                        tool_call_end_pattern = end_pattern
                        tool_call_buffer = tool_call_buffer[tool_call_start + len(start_pattern):]
                        continue

            # If we're in a tool call block and getting more text
            if in_tool_call:
                end_idx = tool_call_buffer.find(tool_call_end_pattern)
                if end_idx != -1:
                    # Complete block received
                    tool_call_text = tool_call_buffer[:end_idx]
                    tool_call_buffer = tool_call_buffer[end_idx + len(tool_call_end_pattern):]
                    in_tool_call = False

                    parsed = _parse_tool_call(tool_call_start_pattern + tool_call_text + tool_call_end_pattern)
                    if parsed:
                        tool_name, args = parsed

                        if pending_text_before_tool.strip():
                            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': pending_text_before_tool}})}\n\n"
                            pending_text_before_tool = ""

                        yield f"data: {json.dumps({'type': 'content_block_start', 'index': tool_call_index, 'content_block': {'type': 'tool_use', 'name': tool_name, 'id': f'toolu_{tool_call_index}'}})}\n\n"
                        yield f"data: {json.dumps({'type': 'content_block_delta', 'index': tool_call_index, 'delta': {'type': 'tool_use_input_json_delta', 'input_json': json.dumps(args)}})}\n\n"
                        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': tool_call_index})}\n\n"
                        yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
                        tool_call_index += 1

                    if tool_call_buffer:
                        pending_text_before_tool += tool_call_buffer
                        tool_call_buffer = ""
                    continue
                else:
                    continue

        # For non-content chunks or when not in tool call mode, use normal processing
        if not content or not in_tool_call:
            for event in _convert_chunk_to_anthropic_events(chunk, is_first=is_first):
                for line in (f"data: {event}\n\n").split("\n"):
                    if line.strip() and not _is_valid_sse_line(line):
                        continue
                yield f"data: {event}\n\n"
        is_first = False

    yield "data: [DONE]\n\n"

