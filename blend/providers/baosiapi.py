"""baosiapi LLM Provider - OpenAI compatible API."""

import os
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
        model: str = "claude-opus-4-7",
        **kwargs: Any,
    ) -> list[str]:
        """Send streaming chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID to use
            **kwargs: Additional parameters

        Returns:
            List of response chunks
        """
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

    def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        url = f"{self._base_url}/models"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        return list(data.get("data", []))
