"""L3 Execution Layer - Dynamic Model Selection based on Complexity and Task Type."""

import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from blend.core.budget import ResourceModel
from blend.core.layers import L3Output
from blend.core.model_config import (
    get_fallback_chain,
    get_gemini_task_types,
    get_model_cost,
    get_model_map,
    load_model_registry,
)
from blend.prompts.strategy import L2_STRATEGY_SYSTEM_TEMPLATE
from blend.providers.base import LLMProvider

# Load from YAML config (cached)
_registry = load_model_registry()

# Backwards compatibility - export these for direct import
MODEL_MAP = get_model_map()
FALLBACK_CHAIN = get_fallback_chain()
MODEL_COST = get_model_cost()
GEMINI_TASK_TYPES = get_gemini_task_types()


@dataclass(frozen=True)
class ModelSelection:
    """Model selection based on complexity and task type."""

    primary: str  # minimax | haiku | sonnet | opus | gemini
    fallback: list[str]
    use_gemini_batch: bool = False


@dataclass(frozen=True)
class LLMOutput:
    """Extended L3 output including finish_reason and tool_calls."""

    content: str
    model_used: str
    tokens_used: int
    tokens_budget_remaining: int
    quality_gate_passed: bool
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None


def _get_provider(model_key: str) -> tuple["LLMProvider", str]:
    """Get provider instance and model name for a model key."""
    from blend.providers import BaosiProvider, LemonProvider, MinimaxProvider

    if model_key not in MODEL_MAP:
        model_key = "haiku"

    provider_class_name, model_name = MODEL_MAP[model_key]

    if provider_class_name == "MinimaxProvider":
        return MinimaxProvider(), model_name  # type: ignore[return-value]
    elif provider_class_name == "LemonProvider":
        return LemonProvider(), model_name  # type: ignore[return-value]
    else:
        return BaosiProvider(), model_name  # type: ignore[return-value]


class Executor:
    """Executes prompts using dynamically selected models."""

    def __init__(self) -> None:
        """Initialize executor with resource model."""
        self.resource_model = ResourceModel()

    def execute(
        self,
        prompt: str,
        complexity: int,
        strategy: dict[str, object] | None = None,
        task_type: str = "general",
    ) -> L3Output:
        """Execute prompt with appropriate model.

        Args:
            prompt: The compressed prompt to execute
            complexity: Complexity score (1-10)
            strategy: Optional L2 strategy for HIGH complexity
            task_type: Task type for model routing

        Returns:
            L3Output with execution results
        """
        # Select model based on complexity, task type, and budget
        selection = self._select_model(complexity, task_type)

        # Try primary model, then fallback chain
        raw_output = None
        model_used = None
        for model_key in [selection.primary] + selection.fallback:
            try:
                raw_output = self._call_model(model=model_key, prompt=prompt, strategy=strategy)
                model_used = model_key
                break
            except Exception:
                continue

        # If all models failed, use minimax as last resort
        if raw_output is None:
            raw_output = self._call_model(model="minimax", prompt=prompt)
            model_used = "minimax"

        # model_used is guaranteed non-None here
        assert model_used is not None
        tokens_used = self._estimate_tokens(raw_output)
        budget = self._get_budget(model_used)
        remaining = max(0, budget - tokens_used)

        return L3Output(
            raw_output=raw_output,
            model_used=model_used,
            tokens_used=tokens_used,
            tokens_budget_remaining=remaining,
            quality_gate_passed=True,
        )

    def execute_messages(
        self,
        messages: list[dict[str, Any]],
        complexity: int,
        strategy: dict[str, object] | None = None,
        task_type: str = "general",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        stop: str | list[str] | None = None,
    ) -> LLMOutput:
        """Execute with full message list and optional tool/format parameters."""
        selection = self._select_model(complexity, task_type)

        raw_output = None
        model_used = None
        finish_reason = "stop"
        tool_calls: list[dict[str, Any]] | None = None

        for model_key in [selection.primary] + selection.fallback:
            try:
                result = self._call_model_messages(
                    model=model_key,
                    messages=messages,
                    strategy=strategy,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop=stop,
                )
                raw_output = result.content
                finish_reason = result.finish_reason
                tool_calls = result.tool_calls
                model_used = model_key
                break
            except Exception:
                continue

        if raw_output is None:
            result = self._call_model_messages(
                model="minimax",
                messages=messages,
                strategy=None,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop,
            )
            raw_output = result.content
            finish_reason = result.finish_reason
            tool_calls = result.tool_calls
            model_used = "minimax"

        assert model_used is not None
        tokens_used = self._estimate_tokens(raw_output)
        budget = self._get_budget(model_used)
        remaining = max(0, budget - tokens_used)

        return LLMOutput(
            content=raw_output,
            model_used=model_used,
            tokens_used=tokens_used,
            tokens_budget_remaining=remaining,
            quality_gate_passed=True,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    def _select_model(
        self,
        complexity: int,
        task_type: str,
    ) -> ModelSelection:
        """Select model based on complexity, task type, and budget.

        Args:
            complexity: Complexity score (1-10)
            task_type: Task type for routing

        Returns:
            ModelSelection with primary and fallback models
        """
        # Check budget availability and apply budget-aware routing
        budget_status = self._check_budget_status()

        # Gemini for hard-core tasks (deep reasoning, tool call, multimodal)
        if task_type in GEMINI_TASK_TYPES:
            if budget_status["gemini"] > 1000:
                return ModelSelection(primary="gemini", fallback=["minimax"])
            return ModelSelection(primary="sonnet", fallback=["haiku", "minimax"])

        # Budget-aware routing - complexity drives model, budget determines availability
        # Thresholds match scorer._determine_tier: LOW≤2, MEDIUM≤5, HIGH≥6
        if complexity <= 2:
            # Tier 1 (complexity 1-2): Haiku primary — 90% Sonnet quality at 1/3 cost
            if budget_status["haiku"] >= 50:
                return ModelSelection(primary="haiku", fallback=[])
            return ModelSelection(primary="minimax", fallback=[])

        if complexity <= 5:
            # Medium complexity: haiku or sonnet
            if budget_status["sonnet"] > 100:
                return ModelSelection(
                    primary="sonnet",
                    fallback=["haiku", "minimax"],
                )
            elif budget_status["haiku"] >= 50:
                return ModelSelection(primary="haiku", fallback=["minimax"])
            return ModelSelection(primary="minimax", fallback=[])

        # High complexity (6-10): sonnet
        if budget_status["sonnet"] > 100:
            return ModelSelection(primary="sonnet", fallback=["haiku", "minimax"])
        elif budget_status["haiku"] >= 50:
            return ModelSelection(primary="haiku", fallback=["minimax"])
        return ModelSelection(primary="minimax", fallback=[])

    def _check_budget_status(self) -> dict[str, int]:
        """Check budget remaining for all models."""
        return {
            "minimax": self.resource_model.get_remaining("minimax"),
            "haiku": self.resource_model.get_remaining("haiku"),
            "sonnet": self.resource_model.get_remaining("sonnet"),
            "opus": self.resource_model.get_remaining("opus"),
            "gemini": self.resource_model.get_remaining("gemini"),
        }

    def _call_model(
        self,
        model: str,
        prompt: str,
        strategy: dict[str, object] | None = None,
    ) -> str:
        """Call the selected model via provider."""
        provider, model_name = _get_provider(model)

        # Inject L2 strategy into prompt if available
        plan = strategy.get("plan") if strategy else None
        if plan and isinstance(plan, (list, tuple)):
            plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
            system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = provider.chat(messages=messages, model=model_name)
        return str(response.content)

    def _call_model_messages(
        self,
        model: str,
        messages: list[dict[str, Any]],
        strategy: dict[str, object] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        stop: str | list[str] | None = None,
    ) -> LLMOutput:
        """Call model with full message list and optional tool/format params."""
        from blend.providers.base import LLMResponse

        provider, model_name = _get_provider(model)

        # Inject L2 strategy as system message
        msgs = list(messages)
        plan = strategy.get("plan") if strategy else None
        if plan and isinstance(plan, (list, tuple)):
            plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
            system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
            msgs = [{"role": "system", "content": system_prompt}] + msgs

        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature != 1.0:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if stop is not None:
            kwargs["stop"] = stop

        response: LLMResponse = provider.chat(messages=msgs, model=model_name, **kwargs)
        return LLMOutput(
            content=str(response.content),
            model_used=model,
            tokens_used=len(str(response.content)) // 4,
            tokens_budget_remaining=0,
            quality_gate_passed=True,
            finish_reason=response.finish_reason if hasattr(response, "finish_reason") else "stop",
            tool_calls=response.tool_calls if hasattr(response, "tool_calls") else None,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(text) // 4

    def _get_budget(self, model: str) -> int:
        """Get token budget for model."""
        budgets = {
            "minimax": 100000,
            "haiku": 200000,
            "sonnet": 200000,
            "opus": 200000,
            "gemini": 200000,
        }
        return budgets.get(model, 200000)

    def stream(
        self,
        prompt: str,
        complexity: int,
        strategy: dict[str, object] | None = None,
        task_type: str = "general",
    ) -> Generator[str, None, None]:
        """Stream prompt execution, yielding text chunks from the provider.

        Selects model, then calls provider.chat_stream() and yields chunks.
        """
        selection = self._select_model(complexity, task_type)

        # Try primary model, then fallback chain
        tried: list[str] = []
        for model_key in [selection.primary] + selection.fallback:
            if model_key in tried:
                continue
            tried.append(model_key)
            try:
                provider, model_name = _get_provider(model_key)
                plan = strategy.get("plan") if strategy else None
                if plan and isinstance(plan, (list, tuple)):
                    plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
                    system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]

                chunks = provider.chat_stream(messages=messages, model=model_name)
                for chunk_json in chunks:
                    try:
                        delta = json.loads(chunk_json)
                        content = ""
                        if isinstance(delta, dict):
                            choices = delta.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                return  # Success
            except Exception:
                continue

        # All models failed - raise
        raise RuntimeError("All model providers failed")

    def stream_messages(
        self,
        messages: list[dict[str, Any]],
        complexity: int,
        strategy: dict[str, object] | None = None,
        task_type: str = "general",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        agent_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        stop: str | list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream with full message list and optional tool/format params.

        Yields dicts with delta content and optional tool_call deltas.
        In agent_mode, L4/L5 are deferred until after stream completes
        (currently no-op since streaming defers post-processing).
        """
        selection = self._select_model(complexity, task_type)

        # Inject L2 strategy as system message
        msgs = list(messages)
        plan = strategy.get("plan") if strategy else None
        if plan and isinstance(plan, (list, tuple)):
            plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
            system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
            msgs = [{"role": "system", "content": system_prompt}] + msgs

        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature != 1.0:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if stop is not None:
            kwargs["stop"] = stop

        tried: list[str] = []
        for model_key in [selection.primary] + selection.fallback:
            if model_key in tried:
                continue
            tried.append(model_key)
            try:
                provider, model_name = _get_provider(model_key)
                chunks = provider.chat_stream(messages=msgs, model=model_name, **kwargs)
                for chunk_json in chunks:
                    try:
                        delta = json.loads(chunk_json)
                        if not isinstance(delta, dict):
                            continue
                        choices = delta.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        result: dict[str, Any] = {"delta": choice.get("delta", {}), "finish_reason": choice.get("finish_reason")}
                        # Forward tool_call deltas
                        if "tool_calls" in choice:
                            result["tool_calls"] = choice["tool_calls"]
                        yield result
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                return  # Success
            except Exception:
                continue

        # All models failed
        raise RuntimeError("All model providers failed")
