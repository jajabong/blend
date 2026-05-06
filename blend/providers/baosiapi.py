"""baosiapi LLM Provider - OpenAI compatible API."""

import os
import time
from collections.abc import Generator
from typing import Any, cast

import httpx

from blend.core.circuit_breaker import get_registry
from blend.providers.base import LLMResponse


def _retry_request(
    fn: Any,
    retries: int = 3,
    provider_name: str = "baosi",
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
        except httpx.HTTPStatusError as e:
            # 401/503 are effectively service down. Trip breaker immediately!
            status_code = e.response.status_code
            if status_code in (401, 503):
                # Force open the circuit by hitting threshold and passing code
                for _ in range(breaker.failure_threshold):
                    breaker.record_failure(error_code=status_code)
            raise
        except Exception:
            raise


class BaosiProvider:
    """Provider for baosiapi (OpenAI-compatible proxy)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize baosiapi provider.

        Args:
            api_key: API key (defaults to ANTHROPIC_API_KEY env)
            base_url: Base URL (defaults to CLAUDE_BASE_URL env)
            timeout: Request timeout in seconds
        """
        self._api_key = api_key or os.environ.get("BAOSI_API_KEY", "")
        self._base_url = base_url or os.environ.get(
            "CLAUDE_BASE_URL", "https://api.baosiapi.com/v1"
        )
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
        model: str = "claude-opus-4-7",
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID to use
            stream: Enable streaming response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with content, model, usage, finish_reason, tool_calls
        """
        url = f"{self._base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        def _do_request() -> dict[str, Any]:
            client = self._get_client()
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return cast(dict[str, Any], resp.json())

        data = _retry_request(_do_request, provider_name="baosi")
        msg = data["choices"][0]["message"]
        return LLMResponse(
            content=msg.get("content", ""),
            model=data["model"],
            usage=data.get("usage", {}),
            raw=data,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            tool_calls=msg.get("tool_calls"),
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-opus-4-7",
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Send streaming chat completion request."""
        url = f"{self._base_url}/chat/completions"

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

        # For stream, we record success on initiation
        registry = get_registry()
        breaker = registry.get("baosi")
        if not breaker.allow_request():
            raise httpx.ConnectError("Circuit breaker open for baosi")

        try:
            yield from _do_stream()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise

    def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        url = f"{self._base_url}/models"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        client = self._get_client()
        response = client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        return list(data.get("data", []))

    def close(self) -> None:
        """Close the persistent HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
