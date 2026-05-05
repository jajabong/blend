"""L3 Execution Layer - Dynamic Model Selection based on Complexity and Task Type."""

import concurrent.futures
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
from blend.providers.base import LLMProvider, LLMResponse

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
    thought: str | None = None


@dataclass(frozen=True)
class RecipeStage:
    """A single stage in a recipe - executed sequentially."""

    model: str  # minimax | haiku | sonnet | opus | gemini | gemini_pro | ...
    role: str  # "draft" | "refine" | "verify" | "enforce" | "execute"
    complexity: int
    timeout: float = 15.0
    strategy_hints: dict[str, Any] | None = None


@dataclass(frozen=True)
class Recipe:
    """A multi-stage execution recipe instead of single model routing."""

    stages: list[RecipeStage]
    ensemble: bool = False
    merge_strategy: str = "best_only"  # "best_only" | "merge" | "vote"


def _get_provider(model_key: str) -> tuple["LLMProvider", str]:
    """Get provider instance and model name for a model key."""
    from blend.providers import BaosiProvider, LemonProvider, MinimaxProvider

    current_map = get_model_map()
    if model_key not in current_map:
        model_key = "haiku"

    provider_class_name, model_name = current_map[model_key]

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
        """Execute prompt with appropriate model. Supports Racing Fallback."""
        selection = self._select_model(complexity, task_type)
        candidates = [selection.primary] + selection.fallback

        def _try_one(m_key: str, is_p: bool) -> L3Output:
            timeout = 120.0 if is_p else 15.0
            response = self._call_model(model=m_key, prompt=prompt, strategy=strategy, timeout=timeout)

            raw_output = response.content
            tokens_used = self._extract_usage(response) or self._estimate_tokens(raw_output)
            budget = self._get_budget(m_key)

            return L3Output(
                raw_output=raw_output,
                model_used=m_key,
                tokens_used=tokens_used,
                tokens_budget_remaining=max(0, budget - tokens_used),
                quality_gate_passed=True,
                thought=getattr(response, "thought", None),
            )

        # Implementation of Racing Fallback
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {}
            # Submit primary
            futures[pool.submit(_try_one, candidates[0], True)] = candidates[0]

            # Wait a short "probe" interval for primary
            done, not_done = concurrent.futures.wait(list(futures.keys()), timeout=3.0)

            if done:
                try:
                    return list(done)[0].result()
                except Exception:
                    pass # Fall through to start fallback

            # Primary is slow or failed, fire second choice if exists
            if len(candidates) > 1:
                futures[pool.submit(_try_one, candidates[1], False)] = candidates[1]

            # Final race between primary and fallback
            while futures:
                done, not_done = concurrent.futures.wait(
                    list(futures.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    try:
                        result = f.result()
                        # Success! Cancel others and return
                        for nf in futures:
                            if nf != f:
                                nf.cancel()
                        return result
                    except Exception:
                        # This candidate failed, remove it
                        del futures[f]
                        # If we have more candidates in the YAML chain, we could add them here
                        # But for now we stick to Top 2 for the race.

                if not futures:
                    break

        # Last resort fallback if everything in the race failed
        # Just use minimax synchronously
        response = self._call_model(model="minimax", prompt=prompt)
        return L3Output(
            raw_output=response.content,
            model_used="minimax",
            tokens_used=self._estimate_tokens(response.content),
            tokens_budget_remaining=0,
            quality_gate_passed=True,
            thought=getattr(response, "thought", None),
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
        """Execute with full message list. Supports Racing Fallback."""
        selection = self._select_model(complexity, task_type)
        candidates = [selection.primary] + selection.fallback

        def _try_one(m_key: str, is_p: bool) -> LLMOutput:
            timeout = 120.0 if is_p else 15.0
            return self._call_model_messages(
                model=m_key,
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
                timeout=timeout,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_try_one, candidates[0], True): candidates[0]}

            # Wait 3s for primary
            done, _ = concurrent.futures.wait(list(futures.keys()), timeout=3.0)
            if done:
                try:
                    return list(done)[0].result()
                except Exception:
                    pass

            if len(candidates) > 1:
                futures[pool.submit(_try_one, candidates[1], False)] = candidates[1]

            while futures:
                done, _ = concurrent.futures.wait(list(futures.keys()), return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    try: return f.result()
                    except Exception: del futures[f]
                if not futures:
                    break

        # Last resort
        return _try_one("minimax", False)

    def _select_model(
        self,
        complexity: int,
        task_type: str,
    ) -> ModelSelection:
        """Select model based on health, complexity, and task type."""
        budget_status = self._check_budget_status()
        current_map = get_model_map()
        current_fallbacks = get_fallback_chain()
        gemini_types = get_gemini_task_types()

        # 1. Determine Initial Intent
        primary = "haiku"
        if task_type in gemini_types:
            primary = "gemini"
        elif task_type == "code":
            primary = "sonnet" if complexity >= 5 else "haiku"
        elif complexity <= 2:
            primary = "haiku" if budget_status["haiku"] > 0 else "minimax"
        elif complexity <= 4:
            primary = "sonnet" if budget_status["sonnet"] > 100 else "haiku"
        else:
            primary = "sonnet"

        # 2. Health-Aware Routing
        from blend.core.circuit_breaker import CircuitState, get_registry
        registry = get_registry()

        def is_healthy(m_key: str) -> bool:
            p_class, _ = current_map.get(m_key, ("BaosiProvider", ""))
            p_name = "baosi" if "Baosi" in p_class else ("lemon" if "Lemon" in p_class else "minimax")
            return registry.get(p_name).state != CircuitState.OPEN

        candidates = [primary] + current_fallbacks.get(primary, [])
        final_primary = "minimax"
        for cand in candidates:
            if is_healthy(cand):
                final_primary = cand
                break

        return ModelSelection(
            primary=final_primary,
            fallback=current_fallbacks.get(final_primary, ["minimax"])
        )

    def _check_budget_status(self) -> dict[str, int]:
        """Check budget remaining for all models."""
        return {
            "minimax": self.resource_model.get_remaining("minimax"),
            "haiku": self.resource_model.get_remaining("haiku"),
            "sonnet": self.resource_model.get_remaining("sonnet"),
            "opus": self.resource_model.get_remaining("opus"),
            "gemini": self.resource_model.get_remaining("gemini"),
        }

    def _select_recipe(
        self,
        complexity: int,
        task_type: str,
        strategy_hints: dict[str, Any] | None = None,
    ) -> Recipe:
        """Select a multi-stage recipe based on complexity and task type.

        Recipe replaces single-model routing with multi-stage execution:
        - LOW (1-2): single stage, direct execute
        - MEDIUM (3-5): draft + refine
        - HIGH (6+): draft + refine + verify
        """
        budget_status = self._check_budget_status()
        gemini_types = get_gemini_task_types()

        stages: list[RecipeStage] = []

        # Determine draft model (fast, cheap)
        draft_model = "haiku" if budget_status.get("haiku", 0) > 0 else "minimax"

        # Gemini task types get gemini as primary
        if task_type in gemini_types:
            stages.append(RecipeStage(model="gemini", role="execute", complexity=min(complexity, 5)))
            return Recipe(stages=stages)

        if complexity <= 2:
            # LOW: single stage
            stages.append(RecipeStage(
                model=draft_model,
                role="execute",
                complexity=complexity,
                timeout=10.0,
            ))
        elif complexity <= 5:
            # MEDIUM: draft + refine
            stages.append(RecipeStage(
                model=draft_model,
                role="draft",
                complexity=1,
                timeout=8.0,
            ))
            refine_model = "sonnet" if budget_status.get("sonnet", 0) > 100 else "haiku"
            stages.append(RecipeStage(
                model=refine_model,
                role="refine",
                complexity=complexity,
                timeout=20.0,
                strategy_hints=strategy_hints,
            ))
        else:
            # HIGH: draft + refine + verify
            stages.append(RecipeStage(
                model=draft_model,
                role="draft",
                complexity=1,
                timeout=8.0,
            ))
            refine_model = "sonnet" if budget_status.get("sonnet", 0) > 100 else "haiku"
            stages.append(RecipeStage(
                model=refine_model,
                role="refine",
                complexity=complexity,
                timeout=30.0,
                strategy_hints=strategy_hints,
            ))
            stages.append(RecipeStage(
                model="gemini",
                role="verify",
                complexity=3,
                timeout=15.0,
            ))

        return Recipe(stages=stages)

    def _execute_recipe(
        self,
        recipe: Recipe,
        prompt: str,
        task_type: str = "general",
    ) -> L3Output:
        """Execute a recipe, running each stage sequentially."""
        draft_output: LLMResponse | None = None
        current_prompt = prompt
        last_model_used = "unknown"
        total_tokens = 0
        final_response: LLMResponse | None = None

        for stage in recipe.stages:
            if stage.role == "draft":
                draft_output = self._call_model(
                    model=stage.model,
                    prompt=f"Provide a detailed technical outline/draft for: {prompt}",
                    strategy=stage.strategy_hints,
                    timeout=stage.timeout,
                )
                total_tokens += self._extract_usage(draft_output) or 0

            elif stage.role == "refine":
                if draft_output is not None:
                    current_prompt = f"User Goal: {prompt}\n\nExisting Draft (Review and finalize):\n{draft_output.content}"
                final_response = self._call_model(
                    model=stage.model,
                    prompt=current_prompt,
                    strategy=stage.strategy_hints,
                    timeout=stage.timeout,
                )
                total_tokens += self._extract_usage(final_response) or self._estimate_tokens(final_response.content)
                last_model_used = stage.model

            elif stage.role == "execute":
                final_response = self._call_model(
                    model=stage.model,
                    prompt=current_prompt,
                    strategy=stage.strategy_hints,
                    timeout=stage.timeout,
                )
                total_tokens += self._extract_usage(final_response) or self._estimate_tokens(final_response.content)
                last_model_used = stage.model

            elif stage.role == "verify":
                final_response = self._call_model(
                    model=stage.model,
                    prompt=f"Quality check: {final_response.content if final_response else current_prompt}",
                    strategy=None,
                    timeout=stage.timeout,
                )
                last_model_used = stage.model

        return L3Output(
            raw_output=final_response.content if final_response else current_prompt,
            model_used=last_model_used,
            tokens_used=total_tokens,
            tokens_budget_remaining=0,
            quality_gate_passed=True,
            thought=getattr(final_response, "thought", None) if final_response else None,
        )

    def _call_model(
        self,
        model: str,
        prompt: str,
        strategy: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Call the selected model via provider, returning full LLMResponse."""
        provider, model_name = _get_provider(model)
        if timeout and hasattr(provider, '_timeout'):
            provider._timeout = timeout

        plan = strategy.get("plan") if strategy else None
        if plan and isinstance(plan, (list, tuple)):
            plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
            system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        return provider.chat(messages=messages, model=model_name)

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
        timeout: float | None = None,
    ) -> LLMOutput:
        """Call selected model with messages and full parameter suite."""
        provider, model_name = _get_provider(model)
        if timeout and hasattr(provider, '_timeout'):
            provider._timeout = timeout

        msgs = list(messages)
        plan = strategy.get("plan") if strategy else None
        if plan and isinstance(plan, (list, tuple)):
            plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
            system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
            msgs = [{"role": "system", "content": system_prompt}] + msgs

        kwargs: dict[str, Any] = {}
        if tools: kwargs["tools"] = tools
        if tool_choice: kwargs["tool_choice"] = tool_choice
        if response_format: kwargs["response_format"] = response_format
        if max_tokens is not None: kwargs["max_tokens"] = max_tokens
        if temperature != 1.0: kwargs["temperature"] = temperature
        if top_p is not None: kwargs["top_p"] = top_p
        if presence_penalty is not None: kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None: kwargs["frequency_penalty"] = frequency_penalty
        if stop is not None: kwargs["stop"] = stop

        response: LLMResponse = provider.chat(messages=msgs, model=model_name, **kwargs)
        return LLMOutput(
            content=str(response.content),
            model_used=model,
            tokens_used=self._extract_usage(response) or len(str(response.content)) // 4,
            tokens_budget_remaining=0,
            quality_gate_passed=True,
            finish_reason=response.finish_reason if hasattr(response, "finish_reason") else "stop",
            tool_calls=response.tool_calls if hasattr(response, "tool_calls") else None,
            thought=getattr(response, "thought", None),
        )

    def _extract_usage(self, response: LLMResponse) -> int | None:
        """Extract completion token count from LLMResponse."""
        usage = response.usage
        if not usage or not isinstance(usage, dict): return None
        return usage.get("completion_tokens") or usage.get("total_tokens")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def _get_budget(self, model: str) -> int:
        """Get token budget from ResourceModel with default fallbacks."""
        budget = self.resource_model.get_budget(model)
        if isinstance(budget, int) and budget > 0: return budget
        fallbacks = {"minimax": 100000000, "haiku": 1000000, "sonnet": 1000000, "opus": 500000, "gemini": 200000}
        return fallbacks.get(model, 200000)

    def stream(
        self,
        prompt: str,
        complexity: int,
        strategy: dict[str, object] | None = None,
        task_type: str = "general",
    ) -> Generator[str, None, None]:
        """Stream prompt execution with sequential fallback."""
        selection = self._select_model(complexity, task_type)
        tried: list[str] = []
        for model_key in [selection.primary] + selection.fallback:
            if model_key in tried: continue
            tried.append(model_key)
            try:
                provider, model_name = _get_provider(model_key)
                plan = strategy.get("plan") if strategy else None
                if plan and isinstance(plan, (list, tuple)):
                    plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
                    system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                else:
                    messages = [{"role": "user", "content": prompt}]

                chunks = provider.chat_stream(messages=messages, model=model_name)
                for chunk_json in chunks:
                    try:
                        delta = json.loads(chunk_json)
                        content = ""
                        if isinstance(delta, dict):
                            choices = delta.get("choices", [])
                            if choices: content = choices[0].get("delta", {}).get("content", "")
                        if content: yield content
                    except: continue
                return
            except: continue
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
        """Stream messages with sequential fallback."""
        selection = self._select_model(complexity, task_type)
        msgs = list(messages)
        plan = strategy.get("plan") if strategy else None
        if plan and isinstance(plan, (list, tuple)):
            plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
            system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
            msgs = [{"role": "system", "content": system_prompt}] + msgs

        kwargs: dict[str, Any] = {}
        if tools: kwargs["tools"] = tools
        if tool_choice: kwargs["tool_choice"] = tool_choice
        if response_format: kwargs["response_format"] = response_format
        if max_tokens is not None: kwargs["max_tokens"] = max_tokens
        if temperature != 1.0: kwargs["temperature"] = temperature
        if top_p is not None: kwargs["top_p"] = top_p
        if presence_penalty is not None: kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None: kwargs["frequency_penalty"] = frequency_penalty
        if stop is not None: kwargs["stop"] = stop

        tried: list[str] = []
        for model_key in [selection.primary] + selection.fallback:
            if model_key in tried: continue
            tried.append(model_key)
            try:
                provider, model_name = _get_provider(model_key)
                chunks = provider.chat_stream(messages=msgs, model=model_name, **kwargs)
                for chunk_json in chunks:
                    try:
                        delta = json.loads(chunk_json)
                        if not isinstance(delta, dict): continue
                        choices = delta.get("choices", [])
                        if not choices: continue
                        choice = choices[0]
                        result: dict[str, Any] = {"delta": choice.get("delta", {}), "finish_reason": choice.get("finish_reason")}
                        if "tool_calls" in choice: result["tool_calls"] = choice["tool_calls"]
                        yield result
                    except: continue
                return
            except: continue
        raise RuntimeError("All model providers failed")
