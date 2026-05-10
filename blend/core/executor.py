"""L3 Execution Layer - Dynamic Model Selection based on Complexity and Task Type."""

from __future__ import annotations

import concurrent.futures
import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from blend.core.budget import ResourceModel
from blend.core.layers import L3Output
from blend.core.model_config import (
    get_complexity_thresholds,
    get_fallback_chain,
    get_gemini_task_types,
    get_model_cost,
    get_model_map,
    load_model_registry,
)
from blend.core.verifier import QualityVerifier
from blend.prompts.strategy import L2_STRATEGY_SYSTEM_TEMPLATE
from blend.providers.base import LLMProvider, LLMResponse

# Scheduler imports for Advisor-Judge integration
from blend.scheduler import (
    get_gemini_queue,
    get_minimax_dispatcher,
    get_quota_alert,
    get_token_filler,
)

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


def _get_provider(model_key: str) -> tuple[LLMProvider, str]:
    """Get provider instance and model name for a model key.

    Uses ProviderPool for instance reuse and connection pooling.
    """
    from blend.providers.pool import get_provider_pool

    current_map = get_model_map()
    if model_key not in current_map:
        model_key = "haiku"

    provider_class_name, model_name = current_map[model_key]
    pool = get_provider_pool()
    return pool.get(model_key, provider_class_name, model_name)


class Executor:
    """Executes prompts using dynamically selected models."""

    def __init__(self) -> None:
        """Initialize executor with resource model and verifier."""
        self.resource_model = ResourceModel()
        self._verifier = QualityVerifier()
        # Scheduler integration for Advisor-Judge architecture
        self._dispatcher = get_minimax_dispatcher()
        self._gemini_queue = get_gemini_queue()
        self._token_filler = get_token_filler()
        self._quota_alert = get_quota_alert()

    def cleanup(self) -> None:
        """Close all pooled provider connections."""
        from blend.providers.pool import get_provider_pool
        get_provider_pool().close_all()

    def _get_quality_level(self, complexity: int) -> str:
        """Convert complexity score to quality level string."""
        thresholds = get_complexity_thresholds()
        if complexity > thresholds["medium_max"]:
            return "HIGH"
        elif complexity > thresholds["low_max"]:
            return "MEDIUM"
        return "LOW"

    def _verify_output(self, output: str, quality_level: str) -> bool:
        """Quick structural verification of output.

        Returns True if output passes basic quality checks.
        """
        result = self._verifier.verify(
            output=output,
            quality_level=quality_level,
            layer_path="L1>L3>L5",
            skip_p0_check=False,
        )
        return result.passed

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

        # Correctness-first racing: verify first result before returning
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {}
            # Submit primary
            futures[pool.submit(_try_one, candidates[0], True)] = candidates[0]

            # Wait up to 3s for primary
            done, not_done = concurrent.futures.wait(list(futures.keys()), timeout=3.0)

            if done:
                try:
                    result = list(done)[0].result()
                    # Correctness check: verify output before returning
                    thresholds = get_complexity_thresholds()
                    if complexity > thresholds["medium_max"]:
                        quality_level = "HIGH"
                    elif complexity > thresholds["low_max"]:
                        quality_level = "MEDIUM"
                    else:
                        quality_level = "LOW"
                    if self._verify_output(result.raw_output, quality_level):
                        return result
                    # Verification failed - wait for fallback
                except Exception:
                    pass

            # Primary failed or verification failed, fire fallback
            if len(candidates) > 1:
                futures[pool.submit(_try_one, candidates[1], False)] = candidates[1]

            # Wait for first successful verification
            while futures:
                done, not_done = concurrent.futures.wait(
                    list(futures.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for f in done:
                    try:
                        result = f.result()
                        # Verify before returning
                        quality_level = self._get_quality_level(complexity)
                        if self._verify_output(result.raw_output, quality_level):
                            # Cancel others
                            for nf in futures:
                                if nf != f:
                                    nf.cancel()
                            return result
                        # This one failed verification, try others
                        del futures[f]
                    except Exception:
                        del futures[f]

                if not futures:
                    break

        # Last resort fallback if everything failed or verification failed
        try:
            response = self._call_model(model="minimax", prompt=prompt)
            return L3Output(
                raw_output=response.content,
                model_used="minimax",
                tokens_used=self._estimate_tokens(response.content),
                tokens_budget_remaining=0,
                quality_gate_passed=True,
                thought=getattr(response, "thought", None),
            )
        except Exception as e:
            # MERCY GATE: Final attempt to provide something
            return L3Output(
                raw_output=f"# --- QUALITY WARNING ---\nThe system is under high load and could not verify the final quality. Here is the last available output attempt:\n\nError: {str(e)}",
                model_used="emergency_fallback",
                tokens_used=0,
                tokens_budget_remaining=0,
                quality_gate_passed=False,
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

        quality_level = self._get_quality_level(complexity)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_try_one, candidates[0], True): candidates[0]}

            # Wait up to 3s for primary
            done, not_done = concurrent.futures.wait(list(futures.keys()), timeout=3.0)
            if done:
                try:
                    result = list(done)[0].result()
                    # Correctness check: verify output before returning
                    if self._verify_output(str(result.content), quality_level):
                        return result
                except Exception:
                    pass

            if len(candidates) > 1:
                futures[pool.submit(_try_one, candidates[1], False)] = candidates[1]

            # Wait for first successful verification
            while futures:
                done, not_done = concurrent.futures.wait(list(futures.keys()), return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    try:
                        result = f.result()
                        if self._verify_output(str(result.content), quality_level):
                            # Cancel others
                            for nf in futures:
                                if nf != f:
                                    nf.cancel()
                            return result
                        del futures[f]
                    except Exception:
                        del futures[f]
                if not futures:
                    break

        # Last resort - skip tools for minimax if verification fails
        try:
            if tools:
                return self._call_model_messages(
                    model="minimax",
                    messages=messages,
                    strategy=strategy,
                    tools=None,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop=stop,
                    timeout=15.0,
                )
            return _try_one("minimax", False)
        except Exception as e:
            # MERCY GATE
            return LLMOutput(
                content=f"# --- QUALITY WARNING ---\nSystem failure. {str(e)}",
                model_used="emergency_fallback",
                tokens_used=0,
                tokens_budget_remaining=0,
                quality_gate_passed=False,
            )

    def _select_model(
        self,
        complexity: int,
        task_type: str,
    ) -> ModelSelection:
        """Select model based on Advisor-Judge architecture.

        三分层流规则：
        - LOW: 极简轻任务 → Minimax执行层
        - MEDIUM: 中等/长文 → Gemini催化剂
        - HIGH: 高难度 → Gemini初稿 + Claude终审

        Claude角色限定：
        - 不做苦力（不承接基础生成）
        - 仅做指导+纠错+终审
        - Token消耗最小化
        """
        from blend.core.circuit_breaker import CircuitState, get_registry
        from blend.core.model_config import (
            get_advisor_judge_models,
            is_high_complexity,
            is_low_complexity,
            is_medium_complexity,
        )

        registry = get_registry()
        current_map = get_model_map()
        current_fallbacks = get_fallback_chain()
        advisor_judge_models = get_advisor_judge_models()

        def is_healthy(m_key: str) -> bool:
            p_class, _ = current_map.get(m_key, ("BaosiProvider", ""))
            p_name = "baosi" if "Baosi" in p_class else ("lemon" if "Lemon" in p_class else "minimax")
            return registry.get(p_name).state != CircuitState.OPEN

        # ========== Advisor-Judge 三分层流 ==========

        if is_low_complexity(complexity):
            # LOW: 极简轻任务 → Minimax执行层
            # 每次尽量拉满4096 Token，不浪费调用次数
            primary = "minimax"
        elif is_medium_complexity(complexity):
            # MEDIUM: 中等/长文/多模态 → Gemini催化剂
            # 单次吃满Token，攒任务批量处理
            if task_type in get_gemini_task_types():
                primary = "gemini_pro_ultra"  # 深度推理/多模态用高级版
            else:
                primary = "gemini_pro"
        else:  # HIGH
            # HIGH: 高难度任务
            # 第一优先：Gemini做初稿
            # Claude仅在终审/纠错时调用（不在这里选，由caller主动调用）
            primary = "gemini_pro_ultra"

        # Health-Aware Fallback
        candidates = [primary] + current_fallbacks.get(primary, [])
        final_primary = "minimax"
        for cand in candidates:
            if is_healthy(cand):
                final_primary = cand
                break

        # Dispatcher-aware fallback: if minimax is selected but rate-limited, use haiku
        if final_primary == "minimax" and not self._dispatcher.can_dispatch():
            fallback_candidates = ["haiku"] + current_fallbacks.get("haiku", [])
            for cand in fallback_candidates:
                if is_healthy(cand):
                    final_primary = cand
                    break

        # Fallback chain: 严格按角色分层
        if is_low_complexity(complexity):
            # LOW: 仅Minimax，不耗用其他额度
            fallback = []
        elif is_medium_complexity(complexity):
            # MEDIUM: Gemini → Minimax兜底
            fallback = ["minimax"]
        else:  # HIGH
            # HIGH: Gemini → Claude(Advisor) → Minimax
            fallback = ["claude_sonnet", "minimax"]

        return ModelSelection(
            primary=final_primary,
            fallback=fallback,
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
        - LOW: single stage, direct execute
        - MEDIUM: draft + refine
        - HIGH: draft + refine + verify
        """
        from blend.core.model_config import (
            is_high_complexity,
            is_low_complexity,
            is_medium_complexity,
        )

        budget_status = self._check_budget_status()
        gemini_types = get_gemini_task_types()

        stages: list[RecipeStage] = []

        # Determine draft model (fast, cheap)
        # Use haiku if minimax is rate-limited, otherwise minimax
        if not self._dispatcher.can_dispatch() or budget_status.get("haiku", 0) > 0:
            draft_model = "haiku"
        else:
            draft_model = "minimax"

        # Gemini task types get gemini as primary
        if task_type in gemini_types:
            stages.append(RecipeStage(model="gemini", role="execute", complexity=min(complexity, 5)))
            return Recipe(stages=stages)

        if is_low_complexity(complexity):
            # LOW: single stage (使用 Minimax，省钱)
            stages.append(RecipeStage(
                model="minimax",
                role="execute",
                complexity=complexity,
                timeout=10.0,
            ))
        elif is_medium_complexity(complexity):
            # MEDIUM: draft + refine (使用 Minimax 生成草稿，省钱)
            stages.append(RecipeStage(
                model="minimax",
                role="draft",
                complexity=1,
                timeout=8.0,
            ))
            refine_model = "sonnet" if budget_status.get("sonnet", 0) > 100 else "minimax"
            stages.append(RecipeStage(
                model=refine_model,
                role="refine",
                complexity=complexity,
                timeout=20.0,
                strategy_hints=strategy_hints,
            ))
        else:  # HIGH
            # HIGH: draft + refine + verify (使用 Minimax 生成草稿，省钱)
            stages.append(RecipeStage(
                model="minimax",
                role="draft",
                complexity=1,
                timeout=8.0,
            ))
            refine_model = "sonnet" if budget_status.get("sonnet", 0) > 100 else "minimax"
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
                verify_content = final_response.content if final_response else current_prompt
                draft_line = f"Original Draft for reference:\n{draft_output.content}" if draft_output else ""
                verify_prompt = f"""Quality check of the refined output.

Review the final output for:
1. Correctness and completeness
2. Security (no eval, exec, dangerous patterns)
3. Whether the draft's key points were incorporated

Final Output:
{verify_content}

{draft_line}
"""
                final_response = self._call_model(
                    model=stage.model,
                    prompt=verify_prompt,
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
        # Token filling for efficiency
        prompt = self._token_filler.fill_prompt(prompt, model)

        # Rate-limit check for MiniMax
        if model == "minimax" and not self._dispatcher.can_dispatch():
            raise RuntimeError("Minimax rate limited")

        provider, model_name = _get_provider(model)
        if timeout and hasattr(provider, "_timeout"):
            provider._timeout = timeout

        plan = strategy.get("plan") if strategy else None
        if plan and isinstance(plan, (list, tuple)):
            plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
            system_prompt = L2_STRATEGY_SYSTEM_TEMPLATE.format(plan_text=plan_text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = provider.chat(messages=messages, model=model_name)

        # Quota tracking for monitoring
        if model == "minimax":
            self._quota_alert.check_and_alert(
                model, self._dispatcher.remaining_calls, self._dispatcher.MAX_CALLS,
            )
        elif model.startswith("gemini"):
            self._quota_alert.check_and_alert(
                model, self._gemini_queue.remaining_calls, self._gemini_queue.MAX_CALLS,
            )

        return response

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
        # Rate-limit check for MiniMax
        if model == "minimax" and not self._dispatcher.can_dispatch():
            raise RuntimeError("Minimax rate limited")

        provider, model_name = _get_provider(model)
        if timeout and hasattr(provider, "_timeout"):
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

        # Quota tracking for monitoring
        if model == "minimax":
            self._quota_alert.check_and_alert(
                model, self._dispatcher.remaining_calls, self._dispatcher.MAX_CALLS,
            )
        elif model.startswith("gemini"):
            self._quota_alert.check_and_alert(
                model, self._gemini_queue.remaining_calls, self._gemini_queue.MAX_CALLS,
            )

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
        """Estimate token count using tiktoken (cl100k_base) or fallback heuristic.

        Note: tiktoken is a required dependency (not optional), so fallback is
        only used if tiktoken fails at runtime. MiniMax uses the same
        cl100k_base encoding as OpenAI, so tiktoken is accurate for all
        current providers.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback heuristic (rarely triggered - tiktoken is required)
            # CJK (Chinese/Korean/Japanese): ~1.5-2 chars per token
            # English: ~4 chars per token
            has_cjk = any("一" <= c <= "鿿" for c in text)
            if has_cjk:
                # More accurate than len*2: CJK is ~1.5 chars/token
                return max(1, int(len(text) * 0.67))
            return max(1, len(text) // 4)

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
                    except Exception: continue
                return
            except Exception: continue

        # MERCY GATE
        yield "\n\n# --- QUALITY WARNING ---\nAll providers failed."

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
                    except Exception: continue
                return
            except Exception: continue

        # If tools were passed and all providers failed, retry without tools
        # Also convert list content to string since providers don't handle list format well
        if tools:
            kwargs_without_tools = {k: v for k, v in kwargs.items() if k != "tools"}
            tried = []  # Reset tried list for retry loop
            # Convert list content (from Anthropic format) to string
            converted_msgs = []
            for m in msgs:
                msg_copy = dict(m)
                content = msg_copy.get("content")
                if isinstance(content, list):
                    # Extract text from [{"type": "text", "text": "..."}]
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    msg_copy["content"] = "\n".join(text_parts)
                converted_msgs.append(msg_copy)

            for model_key in [selection.primary] + selection.fallback:
                if model_key in tried: continue
                tried.append(model_key)
                try:
                    provider, model_name = _get_provider(model_key)
                    chunks = provider.chat_stream(messages=converted_msgs, model=model_name, **kwargs_without_tools)
                    for chunk_json in chunks:
                        try:
                            delta = json.loads(chunk_json)
                            if not isinstance(delta, dict): continue
                            choices = delta.get("choices", [])
                            if not choices: continue
                            choice = choices[0]
                            result = {"delta": choice.get("delta", {}), "finish_reason": choice.get("finish_reason")}
                            yield result
                        except Exception: continue
                    return
                except Exception: continue

        # MERCY GATE for streaming
        yield {"delta": {"content": "\n\n# --- QUALITY WARNING ---\nAll providers failed. System is in emergency mode."}, "finish_reason": "stop"}
