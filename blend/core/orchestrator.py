"""Blend Orchestrator - Coordinates the 5-Layer Pipeline."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from blend.core.budget import ResourceModel
from blend.core.compression import CompressionTrigger, L4Compressor
from blend.core.enforcer import Enforcer
from blend.core.executor import Executor
from blend.core.layers import L1Output, L2Output, L4Output
from blend.core.strategy import StrategyGenerator
from blend.core.tool_executor import execute_tool_calls
from blend.core.verifier import QualityVerifier
from blend.intent.scorer import ComplexityScorer

if TYPE_CHECKING:
    pass


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
    l4_applied: bool
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_count: int = 0
    tool_loop_iterations: int = 0


class BlendOrchestrator:
    """Orchestrates the 5-layer blend pipeline."""

    def __init__(self) -> None:
        """Initialize orchestrator with all layer components."""
        self.scorer = ComplexityScorer()
        self.executor = Executor()
        self.strategy_gen = StrategyGenerator()
        self.l4_compressor = L4Compressor()
        self.compression_trigger = CompressionTrigger()
        self.verifier = QualityVerifier()
        self.enforcer = Enforcer()
        self.resource_model = ResourceModel()

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
        """Process message list through the 5-layer pipeline.

        Preserves multi-turn and tool-call message structure.
        """
        # Build prompt from messages for scoring
        prompt = self._messages_to_prompt(messages)

        layer_path_parts: list[str] = []

        # L1
        layer_path_parts.append("L1")
        complexity_score = self.scorer.score(prompt)
        complexity = complexity_score.total
        tier = complexity_score.tier
        task_type = complexity_score.task_type

        should_compress, compression_result = self._smart_compress(prompt, complexity)
        if should_compress and compression_result is not None:
            compressed_prompt = compression_result.compressed
            l1_compressed = True
        else:
            compressed_prompt = prompt
            l1_compressed = False

        # L2
        l2_output: L2Output | None = None
        if tier == "HIGH":
            layer_path_parts.append("L2")
            strategy_result = self.strategy_gen.generate(
                prompt=compressed_prompt,
                complexity=complexity,
            )
            l2_output = strategy_result.output
            self.resource_model.track_consumption("opus", l2_output.estimated_tokens)

        # L3
        layer_path_parts.append("L3")

        # --- v1.4 Agentic Tool Execution Loop ---
        max_tool_iterations = 10
        current_messages: list[dict[str, Any]] = list(messages)
        tool_loop_iterations = 0
        total_tool_calls = 0
        l3_output = None

        while tool_loop_iterations < max_tool_iterations:
            l3_output = self.executor.execute_messages(
                messages=current_messages,
                complexity=complexity,
                strategy={"plan": l2_output.plan} if l2_output else None,
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

            # Track usage
            self.resource_model.track_consumption(
                l3_output.model_used, l3_output.tokens_used
            )

            # Stop loop if model returned stop or no tool calls
            if l3_output.finish_reason != "tool_calls" or not l3_output.tool_calls:
                break

            # Append assistant message with tool_calls to message history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": l3_output.content or "",
            }
            # Strip internal-only fields before sending back to model
            safe_tool_calls: list[dict[str, Any]] = []
            for tc in l3_output.tool_calls:
                safe_tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    },
                })
            assistant_msg["tool_calls"] = safe_tool_calls
            current_messages.append(assistant_msg)

            # Execute tools and append results
            tool_results = execute_tool_calls(l3_output.tool_calls, tools)
            current_messages.extend(tool_results)
            total_tool_calls += len(l3_output.tool_calls)
            tool_loop_iterations += 1

        # l3_output is guaranteed non-None after loop
        assert l3_output is not None

        # L4
        l4_output: L4Output | None = None
        l4_applied = False
        if self.compression_trigger.should_compress(l3_output.tokens_used, agent_mode=agent_mode):
            layer_path_parts.append("L4")
            l4_applied = True
            l4_result = self.l4_compressor.compress(
                text=l3_output.content,
                original_tokens=l3_output.tokens_used,
            )
            l4_output = L4Output(
                compressed_output=l4_result.compressed_output,
                original_tokens=l4_result.original_tokens,
                compressed_tokens=l4_result.compressed_tokens,
                compression_ratio=l4_result.compression_ratio,
            )

        # L5
        layer_path_parts.append("L5")
        final_output = l4_output.compressed_output if l4_output else l3_output.content
        quality_level = tier

        verification = self.verifier.verify(
            output=final_output,
            quality_level=quality_level,
            layer_path=">".join(layer_path_parts),
            output_tokens=len(final_output) // 4,
            l4_applied=l4_applied,
            task_type=task_type,
            skip_p0_check=agent_mode,
        )

        enforcement = self.enforcer.enforce(
            request={"prompt": prompt},
            layer_path=">".join(layer_path_parts),
            complexity=complexity,
            output_tokens=len(final_output) // 4,
            l4_applied=l4_applied,
            model_used=l3_output.model_used,
        )

        if not enforcement.allowed:
            violations = ", ".join(v.reason for v in enforcement.violations)
            final_output = f"[REJECTED: {violations}]"
            quality_gate_passed = False
        else:
            quality_gate_passed = verification.passed

        layer_path = ">".join(layer_path_parts)

        return OrchestratorResult(
            final_output=final_output,
            layer_path=layer_path,
            complexity=complexity,
            model_used=l3_output.model_used,
            tokens_used=l3_output.tokens_used,
            quality_gate_passed=quality_gate_passed,
            l1_compressed=l1_compressed,
            l4_applied=l4_applied,
            finish_reason=l3_output.finish_reason,
            tool_calls=l3_output.tool_calls,
            tool_call_count=total_tool_calls,
            tool_loop_iterations=tool_loop_iterations,
        )

    def stream_messages(
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
    ) -> Generator[dict[str, Any], None, None]:
        """Stream message list through blend pipeline.

        Preserves multi-turn and tool-call structure.
        L1 and L2 run synchronously; L3 streams real chunks.
        """
        prompt = self._messages_to_prompt(messages)
        layer_path_parts: list[str] = ["L1"]

        # L1: Score complexity
        complexity_score = self.scorer.score(prompt)
        complexity = complexity_score.total
        tier = complexity_score.tier
        task_type = complexity_score.task_type

        # L1 compression
        should_compress, compression_result = self._smart_compress(prompt, complexity)
        if should_compress and compression_result is not None:
            compressed_prompt = compression_result.compressed
            l1_compressed = True
        else:
            compressed_prompt = prompt
            l1_compressed = False

        # L2: Strategy (HIGH complexity only)
        l2_output: L2Output | None = None
        if tier == "HIGH":
            layer_path_parts.append("L2")
            strategy_result = self.strategy_gen.generate(
                prompt=compressed_prompt,
                complexity=complexity,
            )
            l2_output = strategy_result.output
            self.resource_model.track_consumption("opus", l2_output.estimated_tokens)

        layer_path_parts.append("L3")

        strategy_dict: dict[str, object] | None = (
            {"plan": l2_output.plan} if l2_output else None
        )
        for chunk in self.executor.stream_messages(
            messages=messages,
            complexity=complexity,
            strategy=strategy_dict,
            task_type=task_type,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            agent_mode=agent_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
        ):
            layer_path = ">".join(layer_path_parts)
            yield {
                "id": f"chatcmpl-{id(messages)}",
                "choices": [chunk],
                "_blend": {
                    "complexity": complexity,
                    "layer_path": layer_path,
                    "l1_compressed": l1_compressed,
                },
            }

        # Terminal chunk
        yield {
            "id": f"chatcmpl-{id(messages)}",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "_blend": {
                "complexity": complexity,
                "layer_path": ">".join(layer_path_parts),
                "l1_compressed": l1_compressed,
            },
        }

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert message list to a prompt string for scoring/compression."""
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Multi-modal: include text parts, mark others
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(f"{role}: {part.get('text', '')}")
                    else:
                        parts.append(f"[{role}: media content]")
            else:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def process(self, prompt: str) -> OrchestratorResult:
        """Process prompt through all 5 layers.

        Args:
            prompt: User's original prompt

        Returns:
            OrchestratorResult with final output and metadata
        """
        layer_path_parts: list[str] = []

        # ============ L1: Entry Layer ============
        layer_path_parts.append("L1")

        # Step 1a: Score complexity
        complexity_score = self.scorer.score(prompt)
        complexity = complexity_score.total
        tier = complexity_score.tier
        task_type = complexity_score.task_type

        # Step 1b: Smart compression - only compress if worthwhile
        should_compress, compression_result = self._smart_compress(prompt, complexity)
        if should_compress and compression_result is not None:
            compressed_prompt = compression_result.compressed
            l1_compressed = True
            compression_ratio = compression_result.compression_ratio
        else:
            compressed_prompt = prompt
            l1_compressed = False
            compression_ratio = 0.0

        _l1_output = L1Output(
            compressed_prompt=compressed_prompt,
            complexity_score=complexity,
            complexity_breakdown=complexity_score.breakdown,
            route_decision=complexity_score.route_decision,
            l1_compressed=l1_compressed,
            compression_ratio=compression_ratio,
        )

        # ============ L2: Strategy Layer (HIGH complexity only) ============
        l2_output: L2Output | None = None
        if tier == "HIGH":
            layer_path_parts.append("L2")
            strategy_result = self.strategy_gen.generate(
                prompt=compressed_prompt,
                complexity=complexity,
            )
            l2_output = strategy_result.output

            # Track L2 token usage
            self.resource_model.track_consumption("opus", l2_output.estimated_tokens)

        # ============ L3: Execution Layer ============
        layer_path_parts.append("L3")

        # Execute with appropriate model
        l3_output = self.executor.execute(
            prompt=compressed_prompt,
            complexity=complexity,
            strategy={"plan": l2_output.plan} if l2_output else None,
            task_type=task_type,
        )

        # Track usage
        self.resource_model.track_consumption(l3_output.model_used, l3_output.tokens_used)

        # ============ L4: Compression Layer ============
        l4_output: L4Output | None = None
        l4_applied = False

        if self.compression_trigger.should_compress(l3_output.tokens_used):
            layer_path_parts.append("L4")
            l4_applied = True

            l4_result = self.l4_compressor.compress(
                text=l3_output.raw_output,
                original_tokens=l3_output.tokens_used,
            )

            l4_output = L4Output(
                compressed_output=l4_result.compressed_output,
                original_tokens=l4_result.original_tokens,
                compressed_tokens=l4_result.compressed_tokens,
                compression_ratio=l4_result.compression_ratio,
            )

        # ============ L5: Verification Layer ============
        layer_path_parts.append("L5")

        final_output = l4_output.compressed_output if l4_output else l3_output.raw_output

        # Determine quality level for gate selection
        quality_level = tier  # LOW | MEDIUM | HIGH

        # Verify output through quality gates
        verification = self.verifier.verify(
            output=final_output,
            quality_level=quality_level,
            layer_path=">".join(layer_path_parts),
            output_tokens=len(final_output) // 4,
            l4_applied=l4_applied,
            task_type=task_type,
        )

        # ============ Enforce Taboos ============
        enforcement = self.enforcer.enforce(
            request={"prompt": prompt},
            layer_path=">".join(layer_path_parts),
            complexity=complexity,
            output_tokens=len(final_output) // 4,
            l4_applied=l4_applied,
            model_used=l3_output.model_used,
        )

        if not enforcement.allowed:
            # Return rejection output
            violations = ", ".join(v.reason for v in enforcement.violations)
            final_output = f"[REJECTED: {violations}]"
            quality_gate_passed = False
        else:
            quality_gate_passed = verification.passed

        layer_path = ">".join(layer_path_parts)

        return OrchestratorResult(
            final_output=final_output,
            layer_path=layer_path,
            complexity=complexity,
            model_used=l3_output.model_used,
            tokens_used=l3_output.tokens_used,
            quality_gate_passed=quality_gate_passed,
            l1_compressed=l1_compressed,
            l4_applied=l4_applied,
        )

    def stream(self, prompt: str) -> Generator[dict[str, Any], None, None]:
        """Stream prompt through blend pipeline with real provider streaming.

        L1 and L2 run synchronously first (required for complexity scoring
        and strategy injection). L3 streams real chunks from the provider.
        L4 and L5 run on the full accumulated output after stream completes.

        Yields dicts in SSE-compatible format: {"id", "choices": [{"delta": {"content": ...}}]}
        """
        layer_path_parts: list[str] = ["L1"]

        # L1: Score complexity (must complete before L3)
        complexity_score = self.scorer.score(prompt)
        complexity = complexity_score.total
        tier = complexity_score.tier
        task_type = complexity_score.task_type

        # L1 compression
        should_compress, compression_result = self._smart_compress(prompt, complexity)
        if should_compress and compression_result is not None:
            compressed_prompt = compression_result.compressed
            l1_compressed = True
        else:
            compressed_prompt = prompt
            l1_compressed = False

        # L2: Strategy (HIGH complexity only, must complete before L3)
        l2_output: L2Output | None = None
        if tier == "HIGH":
            layer_path_parts.append("L2")
            strategy_result = self.strategy_gen.generate(
                prompt=compressed_prompt,
                complexity=complexity,
            )
            l2_output = strategy_result.output
            self.resource_model.track_consumption("opus", l2_output.estimated_tokens)

        layer_path_parts.append("L3")

        # Stream L3 - yield real chunks from provider
        strategy_dict: dict[str, object] | None = (
            {"plan": l2_output.plan} if l2_output else None
        )
        for chunk in self.executor.stream(
            prompt=compressed_prompt,
            complexity=complexity,
            strategy=strategy_dict,
            task_type=task_type,
        ):
            layer_path = ">".join(layer_path_parts)
            yield {
                "id": f"chatcmpl-{id(prompt)}",
                "choices": [{"delta": {"content": chunk}, "finish_reason": None}],
                "_blend": {
                    "complexity": complexity,
                    "layer_path": layer_path,
                    "l1_compressed": l1_compressed,
                },
            }

        # Stream terminal chunk
        yield {
            "id": f"chatcmpl-{id(prompt)}",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "_blend": {
                "complexity": complexity,
                "layer_path": ">".join(layer_path_parts),
                "l1_compressed": l1_compressed,
            },
        }

    def _smart_compress(
        self, prompt: str, complexity: int
    ) -> tuple[bool, None]:
        """L1 compression removed — always returns False.

        Rationale: L1 compression ROI analysis shows negative ROI on short prompts
        and marginal ROI on long prompts. L4 (output compression) provides
        better token savings. See Phase 1 optimization plan.

        Args:
            prompt: Original prompt (unused)
            complexity: Complexity score (unused)

        Returns:
            Always (False, None) — no compression
        """
        return False, None
