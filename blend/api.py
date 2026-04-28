"""FastAPI application for blend API - OpenAI compatible."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
    from blend.config import get_mcp_servers
    from blend.core.tool_executor import register_mcp_tools

    mcp_servers = get_mcp_servers()
    if mcp_servers:
        register_mcp_tools(mcp_servers)
    yield
    # Cleanup on shutdown if needed


app = FastAPI(
    title="Blend API",
    description="极致成本效率商用 API - 自动智能路由",
    version="2.1.0",
    lifespan=lifespan,  # type: ignore[arg-type]  # lifespan type-stub mismatch with contextlib
)

# Global instances
resource_model = ResourceModel()
orchestrator = BlendOrchestrator()


class Message(BaseModel):
    """Chat message."""

    role: str
    content: str | list[dict[str, Any]]
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
    max_tokens: int
    messages: list[dict[str, Any]]
    stream: bool = False
    system: str | None = None
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


def process_through_layers(prompt: str) -> tuple[str, dict[str, Any]]:
    """Process prompt through blend 5-layer pipeline.

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
        payload = {
            "id": chunk.get("id", chunk_id),
            "choices": chunk.get("choices", []),
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health() -> JSONResponse:
    """Health check — checks provider connectivity and circuit breaker state."""
    import httpx

    status: dict[str, Any] = {
        "status": "healthy",
        "service": "blend",
        "providers": {},
    }
    all_healthy = True

    # Check each provider with a short timeout
    provider_urls = {
        "minimax": "https://api.minimaxi.com",
        "baosi": "https://api.baosiapi.com",
        "lemon": "https://new.lemonapi.site",
    }
    for name, base_url in provider_urls.items():
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{base_url}/health")
                provider_ok = 200 <= resp.status_code < 500
        except Exception:
            provider_ok = False

        status["providers"][name] = "up" if provider_ok else "down"
        if not provider_ok:
            all_healthy = False

    if not all_healthy:
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
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    # Build message list preserving structure (multimodal, tool results, etc.)
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        msg_dict: dict[str, Any] = {"role": m.role, "content": m.content}
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
        payload = {
            "id": chunk.get("id", chunk_id),
            "choices": chunk.get("choices", []),
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


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
    """
    import json as json_mod

    queue: asyncio.Queue[tuple[bool, str | BaseException]] = asyncio.Queue()
    chunk_id_str = f"chatcmpl-{int(time.time() * 1000)}"
    exc_info: BaseException | None = None

    def sync_producer() -> None:
        """Runs in a thread pool — produces chunks into the async queue."""
        nonlocal exc_info
        try:
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
                payload = {
                    "id": chunk.get("id", chunk_id_str),
                    "choices": chunk.get("choices", []),
                }
                queue.put_nowait((False, f"data: {json_mod.dumps(payload)}\n\n"))
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
            raise exc_info
        try:
            is_exc, value = await asyncio.wait_for(queue.get(), timeout=60.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Stream timeout — no chunk received in 60s") from None
        if is_exc:
            raise value
        yield value  # type: ignore[misc]
        if value == "data: [DONE]\n\n":
            break


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List available blend models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "blend",
                "object": "model",
                "created": 1700000000,
                "owned_by": "blend",
                "description": "Intelligent router - auto-selects optimal model",
            },
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
        "layer_architecture": "L1(压缩+评分) > L2(策略,仅HIGH) > L3(执行) > L4(二次压缩) > L5(终审)",
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
    system: str | None,
) -> list[dict[str, Any]]:
    """Convert Anthropic message format to blend internal format.

    - Prepends system prompt as a system role message if provided.
    - Preserves tool_result content blocks from Claude Code.
    """
    result: list[dict[str, Any]] = []

    if system:
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
    choices = chunk.get("choices", [])
    choice = choices[0] if choices else {}
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

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
            "delta": {"stop_reason": finish_reason, "stop_sequence": None},
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
        return StreamingResponse(
            _stream_anthropic(messages, request.tools, request.max_tokens,
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


def _stream_anthropic(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    stop_sequences: list[str] | None,
) -> Generator[str, None, None]:
    """Stream blend output as Anthropic SSE events."""
    is_first = True
    for chunk in orchestrator.stream_messages(
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature or 1.0,
        top_p=top_p,
        stop=stop_sequences,
    ):
        for event in _convert_chunk_to_anthropic_events(chunk, is_first=is_first):
            yield f"data: {event}\n\n"
        is_first = False

