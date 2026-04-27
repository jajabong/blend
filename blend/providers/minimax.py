"""Minimax LLM Provider."""

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


class MinimaxProvider:
    """Provider for Minimax API."""

    BASE_URL = "https://api.minimaxi.com/v1"
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
        """Send streaming chat completion request.

        Args:
            messages: List of message dicts
            model: Model ID to use
            **kwargs: Additional parameters

        Returns:
            List of response chunks
        """
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
