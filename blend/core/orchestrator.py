"""Blend Orchestrator - Coordinates the 4-Layer Pipeline with Draft-Refine logic."""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from blend.core.budget import ResourceModel
from blend.core.enforcer import Enforcer
from blend.core.executor import Executor
from blend.core.strategy import StrategyGenerator
from blend.core.verifier import QualityVerifier
from blend.intent.scorer import ComplexityScorer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorResult:
    """Result of the full orchestration pipeline."""

    final_output: str
    layer_path: str
    complexity: int
    model_used: str
    tokens_used: int
    quality_gate_passed: bool
    l1_compressed: bool
    l4_applied: bool = False  # Deprecated: L4 removed
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_count: int = 0
    tool_loop_iterations: int = 0
    thought: str | None = None


class BlendOrchestrator:

    def __init__(self) -> None:
        self.scorer = ComplexityScorer()
        self.strategy = StrategyGenerator()
        self.strategy_gen = self.strategy # Alias for legacy tests
        self.executor = Executor()
        self.verifier = QualityVerifier()
        self.enforcer = Enforcer()
        self.resource_model = ResourceModel()
        # L0: Semantic Cache for high-frequency engineering tasks
        from blend.core.semantic_cache import SemanticCache
        self.cache = SemanticCache(max_entries=1000)

    def _smart_compress(self, *args: Any, **kwargs: Any) -> bool:
        """Legacy placeholder."""
        return False

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert messages list to prompt string with role prefixes."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Multimodal: each item becomes its own line
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                parts.append(f"{role}: {text}")
                        elif item.get("type") == "image_url":
                            parts.append(f"[{role}: media content]")
            elif isinstance(content, str) and content:
                parts.append(f"{role}: {content}")
            elif content == "" and role:
                parts.append(f"{role}: ")
        return "\n".join(parts)

    def process(self, prompt: str) -> OrchestratorResult:
        """Process a single prompt through the 4-layer pipeline."""
        logger.debug(f"process called with: {prompt}")
        layer_path_parts = ["L1"]
        # 1. L1: Score complexity
        score = self.scorer.score(prompt)
        tier = score.tier
        complexity = score.total
        task_type = score.task_type


        # L0: Semantic Cache check before any execution
        cache_result = self.cache.get(prompt, task_type)
        if cache_result.hit:
            layer_path_parts.extend(["CACHE", "L5"])
            response = cache_result.response or ""
            verification = self.verifier.verify(
                output=response,
                quality_level=tier,
                layer_path=">".join(layer_path_parts),
                output_tokens=len(response) // 4,
                task_type=task_type,
            )
            return OrchestratorResult(
                final_output=response,
                layer_path=">".join(layer_path_parts),
                complexity=complexity,
                model_used=cache_result.model_used or "cached",
                tokens_used=cache_result.tokens_saved,
                quality_gate_passed=verification.passed,
                l1_compressed=False,
            )

        # 2. Strategy generation (HIGH complexity only)
        l2_output = None
        strategy_hints: dict[str, Any] | None = None

        if tier == "HIGH":
            try:
                l2_output = self.strategy.generate(prompt, complexity)
                strategy_hints = {
                    "plan": l2_output.output.plan,
                    "redlines": l2_output.output.quality_redlines,
                    "model_hint": l2_output.output.model_hint,
                }
                layer_path_parts.append("L2")
            except Exception:
                pass  # L2 failed, continue without strategy

        # 3. Recipe-based Execution (multi-stage instead of single model routing)
        # For complexity >= 3: use Recipe (draft + refine + verify)
        # For complexity <= 2: use single-stage execute (routing model)
        if complexity >= 3:
            # Use multi-stage Recipe
            recipe = self.executor._select_recipe(
                complexity=complexity,
                task_type=task_type,
                strategy_hints=strategy_hints,
            )
            l3_output = self.executor._execute_recipe(
                recipe=recipe,
                prompt=prompt,
                task_type=task_type,
            )
            # Build layer path from recipe stages
            for stage in recipe.stages:
                layer_path_parts.append(stage.role.upper())
        else:
            # Use single-stage routing (original behavior)
            l3_output = self.executor.execute(
                prompt=prompt,
                complexity=complexity,
                strategy={"plan": l2_output.output.plan} if l2_output else None,
                task_type=task_type,
            )
            layer_path_parts.append("L3")

        layer_path_parts.append("L5")
        verification = self.verifier.verify(
            output=l3_output.raw_output,
            quality_level=tier,
            layer_path=">".join(layer_path_parts),
            output_tokens=l3_output.tokens_used,
            task_type=task_type,
        )

        final_output = l3_output.raw_output

        # Self-healing loop (up to 1 retry)
        if not verification.passed:
            is_p0_violation = not verification.gates_checked.get("no_p0_vuln", True)

            # If it's a P0 or a quality issue, we try to fix it one last time
            layer_path_parts.append("RETRY")

            # EXPLICIT FEEDBACK: Tell the model EXACTLY what failed
            retry_prompt = f"""Your previous output failed quality validation for the following reasons: {verification.rejection_reason}.
Please rewrite the entire response to fix these issues.
Crucially: If there are security patterns like 'input()' or 'eval()', replace them with safer alternatives or remove them.
User's Original Goal: {prompt}"""

            retry_l3 = self.executor.execute(prompt=retry_prompt, complexity=complexity, task_type=task_type)
            final_output = retry_l3.raw_output

            # Final re-verify
            verification = self.verifier.verify(
                output=final_output,
                quality_level=tier,
                layer_path=">".join(layer_path_parts),
                output_tokens=retry_l3.tokens_used,
                task_type=task_type,
            )

        # Enforcer
        enforcement = self.enforcer.enforce(
            request={"prompt": prompt},
            layer_path=">".join(layer_path_parts),
            complexity=complexity,
            output_tokens=len(final_output) // 4,
            model_used=l3_output.model_used,
        )

        # Final result handling
        if not enforcement.allowed:
            final_output = f"[REJECTED: {', '.join(v.reason for v in enforcement.violations)}]"
            passed = False
        elif not verification.passed:
            # If even after retry it's failing quality...
            is_p0_violation = not verification.gates_checked.get("no_p0_vuln", True)

            if is_p0_violation:
                # Security is non-negotiable
                final_output = f"[REJECTED: Security Vulnerability - {verification.rejection_reason}]"
                passed = False
            else:
                # MERCY LOGIC: For non-P0 issues, deliver with a WARNING
                warning_header = f"""# --- QUALITY WARNING ---
# This response failed some quality gates: {verification.rejection_reason}
# It may contain bugs, incomplete logic, or unsafe coding patterns.
# ------------------------

"""
                final_output = warning_header + final_output
                passed = True # Mark as passed so it's not rejected by the API layer
        else:
            passed = True

        # Store in semantic cache for future hits (high-freq patterns)
        self.cache.set(
            prompt=prompt,
            response=final_output,
            model_used=l3_output.model_used,
            tokens_saved=l3_output.tokens_used,
            task_type=task_type,
        )

        return OrchestratorResult(
            final_output=final_output,
            layer_path=">".join(layer_path_parts),
            complexity=complexity,
            model_used=l3_output.model_used,
            tokens_used=l3_output.tokens_used,
            quality_gate_passed=passed,
            l1_compressed=False,
            thought=getattr(l3_output, "thought", None),
        )

    def process_messages(
        self,
        messages: list[dict[str, Any]],
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
    ) -> OrchestratorResult:
        """Full message-based orchestration with tool-loop and Minimax pre-drafting."""
        layer_path_parts = ["L1"]
        last_msg = messages[-1].get("content", "")
        prompt = last_msg if isinstance(last_msg, str) else str(last_msg)

        score = self.scorer.score(prompt)
        complexity = score.total
        tier = score.tier
        task_type = score.task_type

        # 1. Pre-Drafting (Free brainpower)
        if complexity >= 5 and not agent_mode:
            draft_res = self.executor.execute_messages(
                messages=[{"role": "user", "content": f"Create a technical draft for: {prompt}"}],
                complexity=1,
                task_type="general",
            )
            messages = [{"role": "system", "content": f"Reference Draft (incorporate if useful): {draft_res.content}"}] + messages
            layer_path_parts.append("DRAFT")

        # 2. Strategy
        l2_output = None
        if tier == "HIGH":
            l2_output = self.strategy.generate(prompt, complexity)
            layer_path_parts.append("L2")

        # 3. Execution Loop
        current_messages = list(messages)
        total_tool_calls = 0
        tool_loop_iterations = 0
        l3_output = None

        # Check if client already executed tools (role: "tool" messages present)
        # If so, we'll skip server-side tool execution but keep sending tool messages to the LLM
        # so it can see the tool results and continue the conversation
        client_executed_tools = any(m.get("role") == "tool" for m in messages)

        while tool_loop_iterations < 10:
            l3_output = self.executor.execute_messages(
                messages=current_messages,
                complexity=complexity,
                strategy={"plan": l2_output.output.plan} if l2_output else None,
                task_type=task_type,
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

            current_messages.append({"role": "assistant", "content": l3_output.content, "tool_calls": l3_output.tool_calls})

            if not l3_output.tool_calls:
                break

            # If client already executed tools (sent role: "tool" messages), skip server-side execution
            # and skip verification since client is driving the conversation
            # Also, do NOT return tool_calls to the client - they should have been executed already
            if client_executed_tools:
                logger.debug("Client already executed tools, returning content without tool_calls")
                return OrchestratorResult(
                    final_output=l3_output.content or "",
                    layer_path=">".join(layer_path_parts + ["L3", "L5"]),
                    complexity=complexity,
                    model_used=l3_output.model_used,
                    tokens_used=l3_output.tokens_used,
                    quality_gate_passed=True,
                    l1_compressed=False,
                    finish_reason=l3_output.finish_reason,
                    tool_calls=None,  # Don't send tool_calls back to client
                    tool_call_count=total_tool_calls,
                    tool_loop_iterations=tool_loop_iterations,
                    thought=l3_output.thought,
                )

            # Check if tools are registered for server-side execution
            from blend.core.tool_executor import (
                _INTERNAL_TOOLS,
                are_tools_registered,
                has_internal_tools,
            )
            if not are_tools_registered(tools or []) or has_internal_tools(tools):
                # Tools not registered server-side or are internal tools - filter and return to client
                filtered_tc = None
                if l3_output.tool_calls:
                    # Filter out internal tools (todowrite) from tool_calls
                    filtered_tc = [tc for tc in l3_output.tool_calls
                                   if tc.get("function", {}).get("name") not in _INTERNAL_TOOLS]
                if not filtered_tc:
                    # All tools were filtered out or no tool_calls - return content WITHOUT tool_calls
                    # Override finish_reason to "stop" since we're not returning tool_calls
                    return OrchestratorResult(
                        final_output=l3_output.content or "",
                        layer_path=">".join(layer_path_parts + ["L3", "L5"]),
                        complexity=complexity,
                        model_used=l3_output.model_used,
                        tokens_used=l3_output.tokens_used,
                        quality_gate_passed=True,
                        l1_compressed=False,
                        finish_reason="stop",  # We're returning content, not tool_calls
                        tool_calls=None,
                        tool_call_count=0,
                        tool_loop_iterations=tool_loop_iterations,
                        thought=l3_output.thought,
                    )
                return OrchestratorResult(
                    final_output=l3_output.content,
                    layer_path=">".join(layer_path_parts + ["L3", "TOOL_CALL", "L5"]),
                    complexity=complexity,
                    model_used=l3_output.model_used,
                    tokens_used=l3_output.tokens_used,
                    quality_gate_passed=True,
                    l1_compressed=False,
                    finish_reason=l3_output.finish_reason,
                    tool_calls=filtered_tc,
                    tool_call_count=len(filtered_tc),
                    tool_loop_iterations=tool_loop_iterations,
                    thought=l3_output.thought,
                )

            # Execute tools and add results back to messages
            from blend.core.tool_executor import execute_tool_calls
            tool_results = execute_tool_calls(l3_output.tool_calls, tools)
            current_messages.extend(tool_results)

            total_tool_calls += len(l3_output.tool_calls)
            tool_loop_iterations += 1

        # 4. Verification & Self-Correction
        layer_path_parts.append("L3")
        layer_path_parts.append("L5")

        # l3_output is always set after the execution loop
        assert l3_output is not None
        final_output = l3_output.content
        verification = self.verifier.verify(
            output=final_output,
            quality_level=tier,
            layer_path=">".join(layer_path_parts),
            output_tokens=l3_output.tokens_used,
            task_type=task_type,
            skip_p0_check=agent_mode,
        )
        logger.debug(f"verifier.passed={verification.passed}, reason={verification.rejection_reason}")

        # Retry logic: not passed
        if not verification.passed:
            verification = self.verifier.verify(
                output=final_output,
                quality_level=tier,
                layer_path=">".join(layer_path_parts),
                output_tokens=l3_output.tokens_used,
                task_type=task_type,
                skip_p0_check=agent_mode,
            )
            logger.debug(f"verification.passed is {verification.passed}")

            # Retry logic: not passed
            if not verification.passed:
                logger.debug("entering retry")
                layer_path_parts.append("RETRY")
                # Delta Correction: surgical patch instead of full history resend
                # Extract first 300 chars of original prompt as anchor (avoid sending 200K token history)
                prompt_snippet = prompt[:300] + "..." if len(prompt) > 300 else prompt
                correction_messages = [
                    {"role": "system", "content": f"ORIGINAL GOAL: {prompt_snippet}"},
                    {"role": "assistant", "content": final_output},
                    {"role": "user", "content": f"VERIFICATION FAILED: {verification.rejection_reason}\n\n"
                        f"Only rewrite/fix the specific failing parts. Do NOT repeat or regenerate valid sections."},
                ]
                retry_res = self.executor.execute_messages(messages=correction_messages, complexity=complexity, task_type=task_type)
                final_output = retry_res.content
                verification = self.verifier.verify(output=final_output, quality_level=tier, layer_path=">".join(layer_path_parts), task_type=task_type, skip_p0_check=agent_mode)

        return OrchestratorResult(
            final_output=final_output,
            layer_path=">".join(layer_path_parts),
            complexity=complexity,
            model_used=l3_output.model_used,
            tokens_used=l3_output.tokens_used,
            quality_gate_passed=verification.passed,
            l1_compressed=False,
            finish_reason=l3_output.finish_reason,
            tool_calls=l3_output.tool_calls,
            tool_call_count=total_tool_calls,
            tool_loop_iterations=tool_loop_iterations,
            thought=l3_output.thought,
        )

    def stream_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> Generator[Any, None, None]:
        # 1. Calculate complexity from the last user message
        # Handle list content from tool results: [{"type": "text", "text": "..."}]
        def _extract_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return str(item.get("text", ""))
            return ""

        # Check if client already executed tools (role: "tool" messages present)
        client_executed_tools = any(m.get("role") == "tool" for m in messages)

        # Filter messages for streaming - remove tool results if client executed them
        if client_executed_tools:
            stream_messages = [m for m in messages if m.get("role") != "tool"]
            # Don't pass tools to executor since client already executed them
            stream_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
        else:
            stream_messages = list(messages)
            stream_kwargs = kwargs

        last_user_msg = next((_extract_text(m.get("content", "")) for m in reversed(stream_messages) if m.get("role") == "user"), "")
        score = self.scorer.score(last_user_msg)

        # 2. Generate strategy if high complexity
        strategy: dict[str, Any] | None = None
        if score.tier == "HIGH":
            strategy = {"plan": self.strategy.generate(last_user_msg, score.total).output.plan}

        return self.executor.stream_messages(messages=stream_messages, complexity=score.total, strategy=strategy, **stream_kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Generator[Any, None, None]:
        score = self.scorer.score(prompt)
        strategy: dict[str, Any] | None = None
        if score.tier == "HIGH":
            strategy = {"plan": self.strategy.generate(prompt, score.total).output.plan}
        return self.executor.stream(prompt=prompt, strategy=strategy, **kwargs)
