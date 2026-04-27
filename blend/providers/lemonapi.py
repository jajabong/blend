"""LemonAPI Provider - Gemini models."""

import time
from typing import Any

import httpx

from blend.providers.base import LLMResponse


def _retry_request(fn: Any, retries: int = 3) -> Any:
    """Execute fn with exponential backoff retry on transient errors."""
    for attempt in range(retries):
        try:
            return fn()
        except (httpx.ConnectError, httpx.RemoteProtocolError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


class LemonProvider:
    """Provider for LemonAPI (Gemini models)."""

    BASE_URL = "https://new.lemonapi.site/v1"
    DEFAULT_MODEL = "[L]gemini-3-flash-preview"

    # Available models (must use [L] prefix)
    MODELS = {
        "flash": "[L]gemini-3-flash-preview",
        "flash-thinking": "[L]gemini-2.5-flash-maxthinking",
        "flash-search": "[L]gemini-2.5-flash-search",
        "pro": "[L]gemini-3-pro-preview",
        "pro-thinking": "[L]gemini-2.5-pro-maxthinking",
        "pro-search": "[L]gemini-2.5-pro-search",
        "pro-preview": "[L]gemini-3.1-pro-preview",
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
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return dict[str, Any](resp.json())

        data = _retry_request(_do_request)
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
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                break
                            chunks.append(line[6:])

        _retry_request(_do_stream)
        return chunks
