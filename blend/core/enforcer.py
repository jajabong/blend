"""Enforcement Mechanism - 8 Taboos Auto-Rejection."""

from dataclasses import dataclass

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        """StrEnum fallback for Python < 3.11."""


class TabooType(StrEnum):
    """Taboo violation types."""

    SKIP_L1 = "SKIP_L1"
    OPUS_FORMATTER = "OPUS_FORMATTER"  # Deprecated: L4 removed
    GEMINI_SCATTERED = "GEMINI_SCATTERED"
    L2_OVER_300T = "L2_OVER_300T"
    CROSS_LAYER_JUMP = "CROSS_LAYER_JUMP"
    HARDCODED_CREDENTIALS = "HARDCODE_CREDENTIALS"
    BYPASS_L5 = "BYPASS_L5"


@dataclass(frozen=True)
class Taboo:
    """Defines a taboo rule."""

    id: TabooType
    message: str
    rule: str  # Human-readable rule


# Define all 8 taboos
TABOOS = [
    Taboo(
        id=TabooType.SKIP_L1,
        message="禁止跳过 L1 直接路由 Claude/Gemini",
        rule="IF 请求未经 L1 压缩 → 自动拒绝 + HTTP 401",
    ),
    Taboo(
        id=TabooType.GEMINI_SCATTERED,
        message="禁止 Gemini 零散发（单次需 ≥50% 上限）",
        rule="IF 单次调用 < 50% 上下文上限 → 拒绝 + 归集",
    ),
    Taboo(
        id=TabooType.L2_OVER_300T,
        message="禁止 L2 输出超 300T",
        rule="IF output_tokens > 300 → 截断 + l2_truncated: true + HALT",
    ),
    Taboo(
        id=TabooType.CROSS_LAYER_JUMP,
        message="禁止跨层跳连",
        rule="层与层之间必须按顺序执行",
    ),
    Taboo(
        id=TabooType.HARDCODED_CREDENTIALS,
        message="禁止硬编码凭证",
        rule="所有凭证必须通过环境变量或密钥管理器获取",
    ),
    Taboo(
        id=TabooType.BYPASS_L5,
        message="禁止绕过 L5 质量门",
        rule="所有输出必须经过 L5 质量验证",
    ),
]


@dataclass(frozen=True)
class TabooViolation:
    """Represents a taboo violation."""

    taboo: Taboo
    reason: str


@dataclass(frozen=True)
class EnforcementResult:
    """Result of enforcement check."""

    allowed: bool
    violations: list[TabooViolation]


class Enforcer:
    """Enforces all 8 taboos."""

    def enforce(
        self,
        request: dict[str, object],
        layer_path: str,
        complexity: int | None = None,
        output_tokens: int | None = None,
        l4_applied: bool = False,  # Deprecated: L4 removed, parameter ignored
        gemini_used: bool = False,
        gemini_context_percent: float = 0,
        model_used: str | None = None,
    ) -> EnforcementResult:
        """Enforce all taboos on a request.

        Args:
            request: The incoming request
            layer_path: The layer execution path
            complexity: Complexity score (for determining valid paths)
            output_tokens: Token count of output
            l4_applied: Deprecated. L4 removed in v2.0, always ignored.
            gemini_used: Whether Gemini was used
            gemini_context_percent: Gemini context usage percentage
            model_used: Model that was used for execution

        Returns:
            EnforcementResult with allowed status and violations
        """
        violations: list[TabooViolation] = []

        # Check each taboo
        violations.extend(self._check_l1_required(layer_path))
        violations.extend(self._check_gemini_batch(gemini_used, gemini_context_percent))
        violations.extend(self._check_l4_required(output_tokens, l4_applied))
        violations.extend(self._check_valid_path(layer_path, complexity))
        violations.extend(self._check_l5_required(layer_path))
        violations.extend(self._check_hardcoded_credentials(request))

        allowed = len(violations) == 0

        return EnforcementResult(allowed=allowed, violations=violations)

    def _check_l1_required(self, layer_path: str) -> list[TabooViolation]:
        """Check taboo 1: L1 is required."""
        if "L1" not in layer_path:
            taboo = self._find_taboo(TabooType.SKIP_L1)
            return [TabooViolation(taboo=taboo, reason="Request bypassed L1 entry layer")]
        return []

    def _check_gemini_batch(
        self,
        gemini_used: bool,
        context_percent: float,
    ) -> list[TabooViolation]:
        """Check taboo 3: Gemini batch threshold."""
        if gemini_used and context_percent < 50:
            taboo = self._find_taboo(TabooType.GEMINI_SCATTERED)
            return [
                TabooViolation(
                    taboo=taboo, reason=f"Gemini context usage {context_percent}% < 50% threshold"
                )
            ]
        return []

    def _check_l4_required(
        self,
        output_tokens: int | None,
        l4_applied: bool,
    ) -> list[TabooViolation]:
        """Check taboo 4: L4 required for large outputs.

        Note: L4 has been removed from the pipeline. This check is now a
        no-op that always passes (preserved for backwards compatibility).
        """
        # L4 removed - this check always passes
        return []

    def _check_valid_path(
        self,
        layer_path: str,
        complexity: int | None,
    ) -> list[TabooViolation]:
        """Check taboo 6: Valid layer path."""
        violations = []

        # Check for required L5
        if "L5" not in layer_path:
            taboo = self._find_taboo(TabooType.BYPASS_L5)
            violations.append(
                TabooViolation(taboo=taboo, reason="Request bypassed L5 verification")
            )

        # Check for L2 requirement in HIGH complexity (>= 6 with new thresholds)
        if complexity is not None and complexity >= 6 and "L2" not in layer_path:
            taboo = self._find_taboo(TabooType.CROSS_LAYER_JUMP)
            violations.append(
                TabooViolation(
                    taboo=taboo, reason=f"HIGH complexity (≥6) requires L2 but path is {layer_path}"
                )
            )

        return violations

    def _check_l5_required(self, layer_path: str) -> list[TabooViolation]:
        """Check taboo 8: L5 is required."""
        if "L5" not in layer_path:
            taboo = self._find_taboo(TabooType.BYPASS_L5)
            return [TabooViolation(taboo=taboo, reason="Request bypassed L5 quality gate")]
        return []

    def _check_hardcoded_credentials(self, request: dict[str, object]) -> list[TabooViolation]:
        """Check taboo 7: No hardcoded credentials.

        Only flag actual credential strings, not conversational mentions.
        Detects sk-/pk_-prefixed strings (API key format) and PEM headers.
        """
        violations = []
        request_str = str(request).lower()

        # Actual credential patterns: sk- at start, pem headers, etc.
        # NOT: "api_key", "token", "password" as words (too broad, kills normal discussions)
        credential_patterns = [
            r"sk-[a-z0-9]{20,}",  # sk- followed by 20+ chars (actual API key)
            r"pk_[a-z0-9]{20,}",  # pk_ followed by 20+ chars
            r"-----begin\s+\w+\s+-----",  # PEM private key header
            r"ghp_[a-z0-9]{20,}",  # GitHub personal access token
        ]
        import re

        for pattern in credential_patterns:
            if re.search(pattern, request_str):
                taboo = self._find_taboo(TabooType.HARDCODED_CREDENTIALS)
                violations.append(
                    TabooViolation(
                        taboo=taboo, reason="Hardcoded credential pattern detected in request"
                    )
                )
                break

        return violations

    def _find_taboo(self, taboo_id: TabooType) -> Taboo:
        """Find taboo by ID."""
        for taboo in TABOOS:
            if taboo.id == taboo_id:
                return taboo
        # Fallback
        return Taboo(id=taboo_id, message="Unknown taboo", rule="Unknown")
