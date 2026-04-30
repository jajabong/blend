"""Base provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """Standard LLM response format across all providers."""

    content: str
    model: str
    usage: dict[str, int]
    raw: dict[str, Any]
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    thought: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat request to provider.

        Args:
            messages: List of message dicts [{"role": "user/assistant/system", "content": "..."}]
            model: Optional model override
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content and metadata
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Send streaming chat request.

        Args:
            messages: List of message dicts
            model: Model ID to use
            **kwargs: Additional parameters

        Returns:
            List of response chunk JSON strings
        """
        ...
