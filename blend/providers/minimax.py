"""Minimax LLM Provider."""

import time
from collections.abc import Generator
from typing import Any, cast

import httpx

from blend.core.circuit_breaker import get_registry
from blend.providers.base import LLMResponse


def _retry_request(
    fn: Any,
    retries: int = 3,
    provider_name: str = "minimax",
) -> Any:
    """Execute fn with exponential backoff retry on transient errors.

    Integrates circuit breaker for fast-fail on repeated failures.
    Records success/failure to the circuit breaker.
    """
    registry = get_registry()
    breaker = registry.get(provider_name)

    for attempt in range(retries):
        # Fast-fail if circuit is open
        if not breaker.allow_request():
            raise httpx.ConnectError(f"Circuit breaker open for {provider_name}")

        try:
            result = fn()
            breaker.record_success()
            return result
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException, OSError):
            breaker.record_failure()
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            # Non-transient errors (4xx, 5xx) don't trip circuit breaker
            raise


class MinimaxProvider:
    """Provider for Minimax API."""

    BASE_URL = "https://api.minimax.chat/v1"
    MODEL_ID = "MiniMax-M2.7"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize Minimax provider.

        Args:
            api_key: API key (defaults to MINIMAX_API_KEY env)
            base_url: Base URL (defaults to BASE_URL)
            timeout: Request timeout in seconds
        """
        import os as _os

        self._api_key = api_key or _os.environ.get("MINIMAX_API_KEY", "")
        self._base_url = base_url or self.BASE_URL
        self._timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create a persistent HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID to use (defaults to MiniMax-M2.7)
            stream: Enable streaming response
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content, model, usage, finish_reason, tool_calls
        """
        url = f"{self._base_url}/chat/completions"
        model = model or self.MODEL_ID

        # Inject conciseness constraint to reduce noise/latency
        processed_messages = list(messages)
        has_system = any(m["role"] == "system" for m in processed_messages)
        constraint = "Respond concisely. If providing code, focus on the implementation."
        if not has_system:
            processed_messages.insert(0, {"role": "system", "content": constraint})
        else:
            # Append to existing system prompt
            for m in processed_messages:
                if m["role"] == "system":
                    m["content"] = f"{m['content']}\n{constraint}"
                    break

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model,
            "messages": processed_messages,
            "stream": stream,
            **kwargs,
        }

        def _do_request() -> dict[str, Any]:
            client = self._get_client()
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return cast(dict[str, Any], resp.json())

        data = _retry_request(_do_request, provider_name="minimax")
        msg = data["choices"][0]["message"]
        raw_content = msg.get("content", "")

        # Extract thought process
        content, thought = self._extract_thought(raw_content)

        return LLMResponse(
            content=content,
            model=data["model"],
            usage=data.get("usage", {}),
            raw=data,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            tool_calls=msg.get("tool_calls"),
            thought=thought,
        )

    def _extract_thought(self, content: str) -> tuple[str, str | None]:
        """Extract <think>...</think> content from response.

        Supports multiple blocks and trims whitespace.
        """
        import re

        # Match all think blocks
        thoughts = re.findall(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
        thought_content = "\n---\n".join(t.strip() for t in thoughts) if thoughts else None

        # Remove all think blocks from the main content
        clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

        return clean_content, thought_content

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Send streaming chat completion request."""
        url = f"{self._base_url}/chat/completions"
        model = model or self.MODEL_ID

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        def _do_stream() -> Generator[str, None, None]:
            client = self._get_client()
            with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        yield line[6:]

        # Record success on initiation
        registry = get_registry()
        breaker = registry.get("minimax")
        if not breaker.allow_request():
            raise httpx.ConnectError("Circuit breaker open for minimax")

        try:
            yield from _do_stream()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise

    def close(self) -> None:
        """Close the persistent HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
