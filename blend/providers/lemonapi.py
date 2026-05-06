"""LemonAPI Provider - Gemini models."""

import time
from typing import Any, cast

import httpx

from blend.core.circuit_breaker import get_registry
from blend.providers.base import LLMResponse


def _retry_request(
    fn: Any,
    retries: int = 3,
    provider_name: str = "lemon",
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
            # Trip breaker immediately on 401/503
            status_code = e.response.status_code
            if status_code in (401, 503):
                for _ in range(breaker.failure_threshold):
                    breaker.record_failure(error_code=status_code)
            raise
        except Exception:
            raise


class LemonProvider:
    """Provider for LemonAPI (Gemini models)."""

    BASE_URL = "https://new.lemonapi.site/v1"
    DEFAULT_MODEL = "[L]gemini-3-flash-preview"

    # Available models (Elite Gemini 3 Series)
    MODELS = {
        "flash": "[L]gemini-3-flash-preview",
        "pro": "[L]gemini-3-pro-preview",
        "pro-ultra": "[L]gemini-3.1-pro-preview",
        "image-flash": "[L]gemini-3.1-flash-image-preview",
        "image-pro": "[L]gemini-3-pro-image-preview",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        """Initialize Lemon provider.

        Args:
            api_key: API key (defaults to LEMON_API_KEY env)
            base_url: Base URL (defaults to BASE_URL)
            timeout: Request timeout in seconds
        """
        import os as _os

        self._api_key = api_key or _os.environ.get("LEMON_API_KEY", "")
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
            model: Model ID to use (defaults to [L]gemini-2.5-flash)
            stream: Enable streaming (not recommended for Gemini)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content, model, usage, finish_reason, tool_calls
        """
        url = f"{self._base_url}/chat/completions"
        model = model or self.DEFAULT_MODEL

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

        data = _retry_request(_do_request, provider_name="lemon")
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
        model: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Send streaming chat completion request."""
        url = f"{self._base_url}/chat/completions"
        model = model or self.DEFAULT_MODEL

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

        chunks: list[str] = []

        def _do_stream() -> None:
            client = self._get_client()
            with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        chunks.append(line[6:])

        _retry_request(_do_stream, provider_name="lemon")
        return chunks

    def close(self) -> None:
        """Close the persistent HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
