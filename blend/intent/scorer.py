"""L1 Complexity Scorer - Scores prompt complexity from 1-10."""

from dataclasses import dataclass
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        """StrEnum fallback for Python < 3.11."""


class ComplexityTier(StrEnum):
    """Complexity tier levels."""

    LOW = "LOW"  # 1-2: Minimax direct
    MEDIUM = "MEDIUM"  # 3-5: Haiku/Sonnet
    HIGH = "HIGH"  # 6-10: Opus strategy


class TaskType(StrEnum):
    """Task type for model routing."""

    GENERAL = "general"  # General tasks
    DEEP_REASONING = "deep_reasoning"  # Math, logic, analysis
    TOOL_CALL = "tool_call"  # Tool execution
    MULTIMODAL = "multimodal"  # Image/audio processing
    CODE = "code"  # Code generation


@dataclass(frozen=True)
class ComplexityScore:
    """Result of complexity scoring."""

    total: int  # 1-10
    tier: str  # LOW | MEDIUM | HIGH
    breakdown: dict[str, int]
    route_decision: str  # L3_MINIMAX | L3_HAIKU | L2_OPUS
    task_type: str = TaskType.GENERAL.value  # Task type for routing


class ComplexityScorer:
    """Scores prompt complexity based on multiple dimensions."""

    def score(
        self,
        prompt: str,
        output_length: int = 200,
        creativity: int = 0,
    ) -> ComplexityScore:
        """Score prompt complexity.

        Args:
            prompt: The user prompt to score
            output_length: Expected output length in tokens
            creativity: Creativity requirement (0=none, 1=some, 2=high)

        Returns:
            ComplexityScore with total, tier, breakdown, and route decision
        """
        breakdown = self._score_dimensions(prompt, output_length, creativity)
        total = sum(breakdown.values())

        # Ensure minimum score of 1 for any prompt
        total = max(1, total)

        tier = self._determine_tier(total)
        route = self._determine_route(tier)
        task_type = self._detect_task_type(prompt)

        return ComplexityScore(
            total=total,
            tier=tier,
            breakdown=breakdown,
            route_decision=route,
            task_type=task_type,
        )

    def _score_dimensions(
        self,
        prompt: str,
        output_length: int,
        creativity: int,
    ) -> dict[str, int]:
        """Score each dimension of complexity."""
        steps = self._score_steps(prompt)
        domain = self._score_domain(prompt)
        output = self._score_output_length(output_length)
        creativity_score = max(0, min(2, creativity))
        risk = self._score_risk(prompt)

        return {
            "steps": steps,
            "domain": domain,
            "output": output,
            "creativity": creativity_score,
            "risk": risk,
        }

    def _score_steps(self, prompt: str) -> int:
        """Score based on number of task steps."""
        step_keywords = [
            "first",
            "then",
            "next",
            "after",
            "before",
            "step",
            "sequence",
            "order",
            "流程",
            "步骤",
            "1.",
            "2.",
            "3.",
            "首先",
            "其次",
        ]
        count = sum(1 for kw in step_keywords if kw.lower() in prompt.lower())

        # Check for implicit multi-step patterns
        multi_keywords = ["multi", "several", "multiple", "多个", "各种", "different"]
        if any(kw in prompt.lower() for kw in multi_keywords):
            count += 2

        if count >= 3:
            return 2
        elif count >= 1:
            return 1
        return 0

    def _score_domain(self, prompt: str) -> int:
        """Score based on domain depth."""
        high_risk_keywords = [
            "security",
            "auth",
            "crypto",
            "finance",
            "medical",
            "legal",
            "architecture",
            "distributed",
            "scalability",
            "distributed",
            "system",
            "infrastructure",
            "安全",
            "金融",
            "架构",
            "系统",
            "项目",
        ]
        medium_keywords = [
            "code",
            "api",
            "database",
            "sql",
            "python",
            "javascript",
            "class",
            "function",
            "algorithm",
            "data",
            "test",
            "code",
            "implement",
            "development",
            "process",
            "代码",
            "数据库",
            "算法",
            "开发",
            "处理",
            "检查",
            "审计",
            "分析",
            "评估",
        ]

        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in high_risk_keywords):
            return 2
        elif any(kw in prompt_lower for kw in medium_keywords):
            return 2  # Code tasks are always at least medium complexity
        return 0

    def _score_output_length(self, output_length: int) -> int:
        """Score based on expected output length."""
        if output_length > 500:
            return 2
        elif output_length >= 200:
            return 1
        return 0

    def _score_risk(self, prompt: str) -> int:
        """Score based on potential risk of wrong answer."""
        high_risk_keywords = [
            "must",
            "critical",
            "production",
            "money",
            "death",
            "emergency",
            "important",
            "careful",
            "dangerous",
            "design",
            "architecture",
            "system design",
            "关键",
            "重要",
            "危险",
            "设计",
        ]
        medium_risk_keywords = [
            "should",
            "recommend",
            "suggest",
            "probably",
            "handle",
            "edge case",
            "validation",
            "error",
            "处理",
            "验证",
            "异常",
        ]

        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in high_risk_keywords):
            return 2
        elif any(kw in prompt_lower for kw in medium_risk_keywords):
            return 1
        return 0

    def _determine_tier(self, total: int) -> str:
        """Determine tier from total score."""
        if total <= 2:
            return ComplexityTier.LOW.value
        elif total <= 5:
            return ComplexityTier.MEDIUM.value
        return ComplexityTier.HIGH.value

    def _determine_route(self, tier: str) -> str:
        """Determine routing based on tier."""
        if tier == ComplexityTier.LOW.value:
            return "L3_MINIMAX"
        elif tier == ComplexityTier.MEDIUM.value:
            return "L3_HAIKU"
        return "L2_OPUS"

    def _detect_task_type(self, prompt: str) -> str:
        """Detect task type for model routing.

        Gemini is used for: deep reasoning, tool call, multimodal
        Claude is used for: general, code
        """
        import re

        prompt_lower = prompt.lower()

        # Deep reasoning keywords (checked first - highest priority for Gemini)
        reasoning_keywords = [
            r"\breasoning\b",
            r"\blogic\b",
            r"\bmath\b",
            r"\bcalculate\b",
            r"\bprove\b",
            r"\banalyz[eing]\b",
            r"\banalysis\b",
            r"\bevaluat[eing]\b",
            r"\bcompare\b",
            r"推理",
            r"逻辑",
            r"数学",
            r"计算",
            r"分析",
            r"证明",
        ]
        for kw in reasoning_keywords:
            if re.search(kw, prompt_lower):
                return TaskType.DEEP_REASONING.value

        # Tool call keywords (checked before code - to route Gemini for tool use)
        tool_keywords = [
            # Specific patterns: "call/invoke/execute the tool|function|api"
            r"\b(call|invoke|execute)\s+(the\s+)?(tool|function|api)\b",
            # "get X from/via the tool|api"
            r"\bget\s+\w+\s+(from|via)\s+(the\s+)?(tool|api)\b",
            # "(tool|function) call/execution with"
            r"(tool|function)\s+(call|execution)\s+with\b",
            # "use the tool|function|api"
            r"use\s+(the\s+)?(tool|function|api)\b",
            # Chinese: tool call/execute/invoke
            r"调用\s+\w+\s*(工具|API|函数)",
            r"(调用|执行|使用)\s*(工具|API|函数)",
            r"工具调用",
        ]
        for kw in tool_keywords:
            if re.search(kw, prompt_lower):
                return TaskType.TOOL_CALL.value

        # Multimodal keywords
        multimodal_keywords = [
            r"\bimage\b",
            r"\baudio\b",
            r"\bvideo\b",
            r"\bpicture\b",
            r"\bphoto\b",
            r"图片",
            r"音频",
            r"视频",
            r"图像",
        ]
        for kw in multimodal_keywords:
            if re.search(kw, prompt_lower):
                return TaskType.MULTIMODAL.value

        # Code keywords (checked last - default for Claude)
        code_keywords = [
            r"\bcode\b",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\bjava\b",
            r"\brust\b",
            r"\bgo\b",
            r"\bclass\b",
            r"\bimplement\b",
            r"\bdebug\b",
            r"代码",
            r"编程",
            r"开发",
        ]
        for kw in code_keywords:
            if re.search(kw, prompt_lower):
                return TaskType.CODE.value

        return TaskType.GENERAL.value
