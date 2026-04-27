"""L2 Strategy Layer - Opus Strategy Generation for High Complexity Tasks."""

import json
import re
from dataclasses import dataclass

from blend.core.layers import L2Output
from blend.providers.baosiapi import BaosiProvider

OPUS_SYSTEM_PROMPT = """你是一个高级策略生成器。为给定任务生成结构化执行计划。

要求：
1. plan: 3-5 个执行步骤（每步简洁）
2. quality_redlines: 2-4 个质量红线（安全、正确性、边界）
3. boundary_cases: 2-3 个边界情况
4. model_recommendation: 推荐执行模型（Opus/Sonnet/Haiku，基于复杂度、上下文需求、推理深度）

请返回 JSON 格式：
{"plan": ["步骤1", "步骤2"], "quality_redlines": ["红线1"], "boundary_cases": ["情况1"], "model_recommendation": "Sonnet"}

不要包含任何其他文字，只返回 JSON。"""


@dataclass(frozen=True)
class StrategyResult:
    """Result of strategy generation."""

    output: L2Output
    truncated: bool


class StrategyGenerator:
    """Generates execution strategy for HIGH complexity tasks using Opus."""

    def generate(
        self,
        prompt: str,
        complexity: int,
    ) -> StrategyResult:
        """Generate execution strategy via Opus API or rule-based fallback.

        Args:
            prompt: The original prompt
            complexity: Complexity score (8-10 for L2)

        Returns:
            StrategyResult with L2Output and truncation flag
        """
        plan: list[str]
        redlines: list[str]
        boundary_cases: list[str]
        model_hint: str

        # Try Opus for HIGH complexity (>= 6, matching scorer._determine_tier thresholds)
        if complexity >= 6:
            try:
                plan, redlines, boundary_cases, model_recommendation = self._call_opus(prompt)
                # Use Opus's model recommendation if provided, otherwise fallback to rules
                model_hint = model_recommendation if model_recommendation else self._determine_model_hint(complexity)
            except Exception:
                # Fallback to rule-based on Opus failure
                plan = self._generate_plan(prompt)
                redlines = self._generate_redlines(prompt)
                boundary_cases = self._identify_boundary_cases(prompt)
                model_hint = self._determine_model_hint(complexity)
        else:
            # Low/medium complexity: rule-based only
            plan = self._generate_plan(prompt)
            redlines = self._generate_redlines(prompt)
            boundary_cases = self._identify_boundary_cases(prompt)
            model_hint = self._determine_model_hint(complexity)
        estimated_tokens = self._estimate_tokens(plan, redlines)

        output = L2Output(
            plan=plan,
            quality_redlines=redlines,
            boundary_cases=boundary_cases,
            model_hint=model_hint,
            estimated_tokens=estimated_tokens,
        )

        truncated = estimated_tokens > 300

        return StrategyResult(output=output, truncated=truncated)

    def _call_opus(self, prompt: str) -> tuple[list[str], list[str], list[str], str]:
        """Call Opus via BaosiProvider to generate strategy.

        Returns:
            tuple of (plan, redlines, boundary_cases, model_recommendation)
        """
        provider = BaosiProvider()
        messages = [
            {"role": "system", "content": OPUS_SYSTEM_PROMPT},
            {"role": "user", "content": f"任务：{prompt}"},
        ]

        response = provider.chat(
            messages=messages,
            model="claude-opus-4-7",
        )

        # Parse JSON from response
        content = response.content.strip()

        # Try to extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON from Opus")
        else:
            raise ValueError("No JSON in Opus response")

        plan = data.get("plan", [])
        redlines = data.get("quality_redlines", [])
        boundary_cases = data.get("boundary_cases", [])
        model_recommendation = data.get("model_recommendation", "")

        # Validate and limit
        if not isinstance(plan, list):
            plan = []
        if not isinstance(redlines, list):
            redlines = []
        if not isinstance(boundary_cases, list):
            boundary_cases = []
        if not isinstance(model_recommendation, str):
            model_recommendation = ""

        return plan[:5], redlines[:4], boundary_cases[:3], model_recommendation

    def _generate_plan(self, prompt: str) -> list[str]:
        """Generate execution plan steps."""
        # Analyze prompt for key elements
        plan_steps = []

        # Identify task type
        prompt_lower = prompt.lower()
        if "design" in prompt_lower:
            plan_steps.append("1. Analyze requirements and constraints")
            plan_steps.append("2. Define system boundaries")
            plan_steps.append("3. Design component architecture")
            plan_steps.append("4. Specify interfaces and protocols")
            plan_steps.append("5. Review against requirements")
        elif "build" in prompt_lower or "implement" in prompt_lower:
            plan_steps.append("1. Parse requirements")
            plan_steps.append("2. Design data models")
            plan_steps.append("3. Implement core logic")
            plan_steps.append("4. Add error handling")
            plan_steps.append("5. Write documentation")
        elif "optimize" in prompt_lower:
            plan_steps.append("1. Profile current implementation")
            plan_steps.append("2. Identify bottlenecks")
            plan_steps.append("3. Apply optimization strategies")
            plan_steps.append("4. Verify improvements")
        else:
            plan_steps.append("1. Understand the problem")
            plan_steps.append("2. Break down into components")
            plan_steps.append("3. Implement solution")
            plan_steps.append("4. Validate against requirements")

        return plan_steps[:4]  # Limit to 4 steps for token budget

    def _generate_redlines(self, prompt: str) -> list[str]:
        """Generate quality redlines."""
        redlines = []
        prompt_lower = prompt.lower()

        # Common security redlines
        if any(kw in prompt_lower for kw in ["auth", "login", "user", "password"]):
            redlines.append("No hardcoded credentials or secrets")
        if any(kw in prompt_lower for kw in ["api", "http", "request"]):
            redlines.append("Validate all input parameters")
        if any(kw in prompt_lower for kw in ["data", "database", "storage"]):
            redlines.append("Handle null and empty cases")

        # Always include these
        redlines.append("No injection vulnerabilities")
        redlines.append("Proper error handling")

        return redlines[:3]  # Limit to 3 redlines for token budget

    def _identify_boundary_cases(self, prompt: str) -> list[str]:
        """Identify boundary cases to handle."""
        cases = []
        prompt_lower = prompt.lower()

        if "input" in prompt_lower or "user" in prompt_lower:
            cases.append("Empty input")
            cases.append("Extremely long input")
        if "number" in prompt_lower or "count" in prompt_lower:
            cases.append("Zero and negative values")
            cases.append("Maximum value overflow")
        if "file" in prompt_lower or "upload" in prompt_lower:
            cases.append("Missing file")
            cases.append("Corrupted file")
        if "network" in prompt_lower or "api" in prompt_lower:
            cases.append("Network timeout")
            cases.append("Service unavailable")

        return cases[:3]  # Limit to 3 boundary cases

    def _determine_model_hint(self, complexity: int) -> str:
        """Determine which model should execute."""
        if complexity >= 9:
            return "Opus"
        return "Sonnet"

    def _estimate_tokens(self, plan: list[str], redlines: list[str]) -> int:
        """Estimate output token count."""
        # Rough estimate: ~4 tokens per word
        plan_tokens = sum(len(step.split()) for step in plan) * 4
        redline_tokens = sum(len(r.split()) for r in redlines) * 4
        overhead_tokens = 50  # JSON structure overhead

        return int(plan_tokens + redline_tokens + overhead_tokens)
