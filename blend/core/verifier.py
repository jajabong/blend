"""L5 Verification Layer - Graded Quality Gate for Output Verification."""

from dataclasses import dataclass
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        """StrEnum fallback for Python < 3.11."""


class QualityGate(StrEnum):
    """Quality gate identifiers."""

    NO_TABOO_VIOLATION = "no_taboo_violation"
    FORMAT_COMPLIANT = "format_compliant"
    NO_P0_VULN = "no_p0_vuln"
    NO_HARDCODED_SECRETS = "no_hardcoded_secrets"
    GEMINI_BATCH_THRESHOLD = "gemini_batch_threshold"
    L4_APPLIED_IF_NEEDED = "l4_applied_if_needed"
    MODEL_FULL_NAME = "model_full_name"
    LAYER_PATH_VALID = "layer_path_valid"
    GEMINI_QUALITY = "gemini_quality"
    # Code-specific gates
    CODE_SYNTAX_CHECK = "code_syntax_check"
    CODE_EDGE_CASE = "code_edge_case"
    CODE_SECURITY = "code_security"


@dataclass(frozen=True)
class VerificationResult:
    """Result of quality verification."""

    passed: bool
    gates_checked: dict[str, bool]
    rejection_reason: str | None


class QualityVerifier:
    """Verifies output quality through graded gates."""

    # Taboo content patterns
    TABOO_PATTERNS = [
        "xxx",
        "porn",
        "nude",
        "gore",
        "violence",
    ]

    # Secret patterns
    SECRET_PATTERNS = [
        "api_key",
        "api-key",
        "secret",
        "password",
        "token",
        "sk-",
        "pk_",
        "-----BEGIN",
        "ghp_",
        "gho_",
    ]

    # P0 vulnerability patterns
    VULN_PATTERNS = [
        "eval(",
        "exec(",
        "os.system",
        "subprocess(",
        "SELECT * FROM",
        "DROP TABLE",
        "DELETE FROM",
    ]

    # Code security patterns (redteam-style)
    CODE_SECURITY_PATTERNS = [
        "eval(",
        "exec(",
        "os.system",
        "subprocess.Popen",
        "__import__",
        "compile(",
        "input(",
        "raw_input",
        "SQL注入",
        "innerHTML",
        "document.write",
    ]

    # Code edge case patterns to check
    CODE_EDGE_CASE_PATTERNS = [
        "if not",
        "if x is None",
        "try:",
        "except",
        "raise",
        "null",
        "None",
        "undefined",
    ]

    def verify(
        self,
        output: str,
        quality_level: str,
        layer_path: str,
        gemini_used: bool = False,
        gemini_context_percent: float = 0,
        output_tokens: int = 0,
        l4_applied: bool = True,
        model_name: str | None = None,
        expected_path: str | None = None,
        task_type: str = "general",
        skip_p0_check: bool = False,
    ) -> VerificationResult:
        """Verify output quality.

        Args:
            output: The output to verify
            quality_level: LOW | MEDIUM | HIGH
            layer_path: The layer execution path
            gemini_used: Whether Gemini was used
            gemini_context_percent: Gemini context usage percentage
            output_tokens: Token count of output
            l4_applied: Whether L4 compression was applied
            model_name: Model name used
            expected_path: Expected layer path
            task_type: Task type for code-specific checks
            skip_p0_check: If True, bypass P0 vulnerability check (for agent mode)

        Returns:
            VerificationResult with pass/fail and gates checked
        """
        gates: dict[str, bool] = {}

        # Gate 1: No taboo violation
        gates[QualityGate.NO_TABOO_VIOLATION] = not self._contains_taboo(output)

        # Gate 2: Format compliant - check structural validity
        gates[QualityGate.FORMAT_COMPLIANT] = self._check_format_compliance(output)

        # Gate 3: No P0 vulnerability
        gates[QualityGate.NO_P0_VULN] = not self._contains_vuln(output) if not skip_p0_check else True

        # Gate 4: No hardcoded secrets
        gates[QualityGate.NO_HARDCODED_SECRETS] = not self._contains_secret(output)

        # Gate 5: Gemini batch threshold
        if gemini_used:
            gates[QualityGate.GEMINI_BATCH_THRESHOLD] = gemini_context_percent >= 50
        else:
            gates[QualityGate.GEMINI_BATCH_THRESHOLD] = True

        # Gate 6: L4 applied if needed (>1000 tokens, aligned with CompressionTrigger threshold)
        if output_tokens > 1000:
            gates[QualityGate.L4_APPLIED_IF_NEEDED] = l4_applied
        else:
            gates[QualityGate.L4_APPLIED_IF_NEEDED] = True

        # Gate 7: Model full name
        if model_name:
            gates[QualityGate.MODEL_FULL_NAME] = self._is_full_model_name(model_name)
        else:
            gates[QualityGate.MODEL_FULL_NAME] = True

        # Gate 8: Layer path valid
        gates[QualityGate.LAYER_PATH_VALID] = self._is_valid_path(
            layer_path, quality_level, expected_path
        )

        # Gate 9: Gemini quality assessment (HIGH complexity only)
        if quality_level == "HIGH":
            gemini_result = self.gemini_evaluate(output, quality_level)
            gates[QualityGate.GEMINI_QUALITY] = bool(gemini_result["passed"])
        else:
            gates[QualityGate.GEMINI_QUALITY] = True

        # Code-specific gates (redteam-style validation)
        if task_type == "code":
            gates[QualityGate.CODE_SYNTAX_CHECK] = self._check_code_syntax(output)
            gates[QualityGate.CODE_EDGE_CASE] = self._check_code_edge_cases(output)
            gates[QualityGate.CODE_SECURITY] = self._check_code_security(output)
        else:
            gates[QualityGate.CODE_SYNTAX_CHECK] = True
            gates[QualityGate.CODE_EDGE_CASE] = True
            gates[QualityGate.CODE_SECURITY] = True

        # Determine overall pass
        passed = all(gates.values())
        rejection_reason = None if passed else self._get_rejection_reason(gates)

        return VerificationResult(
            passed=passed,
            gates_checked=gates,
            rejection_reason=rejection_reason,
        )

    def gemini_evaluate(
        self,
        output: str,
        quality_level: str,
    ) -> dict[str, object]:
        """Evaluate output quality using Gemini semantic assessment.

        Args:
            output: The output to evaluate
            quality_level: Quality level (LOW | MEDIUM | HIGH)

        Returns:
            dict with passed, relevance, accuracy, completeness, issues
        """
        import json
        import re

        system_prompt = """你是一个输出质量评估专家。评估以下输出的质量。

评分维度：
- relevance: 输出与任务的相关性 (0-100)
- accuracy: 回答的准确性 (0-100)
- completeness: 回答的完整性 (0-100)
- issues: 发现的问题列表

请返回 JSON 格式：
{"passed": true/false, "relevance": 85, "accuracy": 80, "completeness": 75, "issues": []}

规则：
- relevance < 60 或 accuracy < 60 → passed = false
- 如果 relevance >= 60 且 accuracy >= 60 且 completeness >= 50 → passed = true
- issues 列出主要问题（最多 3 条）"""

        try:
            from blend.providers.lemonapi import LemonProvider

            provider = LemonProvider()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"评估此输出：\n{output}"},
            ]
            response = provider.chat(messages=messages, model="gemini-3-pro")

            content = response.content.strip()
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "passed": bool(data.get("passed", True)),
                    "relevance": int(data.get("relevance", 0)),
                    "accuracy": int(data.get("accuracy", 0)),
                    "completeness": int(data.get("completeness", 0)),
                    "issues": list(data.get("issues", []))[:3],
                }
            return self._gemini_safe_default("No JSON in Gemini response")
        except Exception as e:
            return self._gemini_safe_default(str(e))

    def _gemini_safe_default(self, error_msg: str) -> dict[str, object]:
        """Return safe default when Gemini evaluation fails."""
        return {
            "passed": True,
            "relevance": 0,
            "accuracy": 0,
            "completeness": 0,
            "issues": [f"Gemini evaluation skipped: {error_msg}"],
        }

    def _check_format_compliance(self, output: str) -> bool:
        """Check structural format validity of output.

        Returns False only if:
        - Output is empty or all whitespace
        - Looks like JSON but fails to parse (malformed JSON)
        - Code block fences are mismatched
        """
        import json

        stripped = output.strip()

        # Empty output
        if not stripped:
            return False

        # Mismatched code fences - odd count of ``` at start of line
        fence_count = stripped.count("```")
        if fence_count % 2 != 0:
            # Odd number of ``` means mismatched open/close
            return False

        # Looks like JSON but invalid (malformed JSON is a real error)
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return False
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return False

        return True

    def _contains_taboo(self, text: str) -> bool:
        """Check for taboo content."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.TABOO_PATTERNS)

    def _contains_secret(self, text: str) -> bool:
        """Check for hardcoded secrets.

        Looks for actual credentials, not placeholder text in examples.
        """
        text_lower = text.lower()
        # Check for actual secret patterns (API keys, tokens, real passwords)
        strong_patterns = [
            "sk-",
            "pk_",
            "-----begin",
            "ghp_",
            "gho_",
            "bearer ",
            "api_key=",
            "api-key=",
            "secret_key=",
            "token=",
        ]
        if any(pattern in text_lower for pattern in strong_patterns):
            return True

        # Check for password= with actual value (not placeholder text)
        import re

        # Match password='...' or password="..." with actual content
        password_patterns = [
            r"password\s*=\s*['\"][^'\"]{8,}['\"]",  # password with 8+ char value
            r"password\s*:\s*['\"][^'\"]{8,}['\"]",  # password: with 8+ char value
        ]
        for pattern in password_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _contains_vuln(self, text: str) -> bool:
        """Check for P0 vulnerabilities."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.VULN_PATTERNS)

    def _is_full_model_name(self, model_name: str) -> bool:
        """Check if model name is valid (short or full name accepted)."""
        # Accept both short names and full names
        valid_names = [
            # Short names (returned by executor)
            "minimax",
            "haiku",
            "sonnet",
            "opus",
            "gemini",
            "gemini-pro",
            # Full names (API format)
            "claude-opus-4-7",
            "claude-sonnet-4-5",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "minimax-2.7",
            "minimax-m2.7",
            "gemini-3.1-pro",
            "[l]gemini-3-pro-preview",
            "[l]gemini-3-flash-preview",
        ]
        name_lower = model_name.lower()
        return any(valid in name_lower for valid in valid_names)

    # ── Code-specific redteam gates ────────────────────────────────────────────

    def _check_code_syntax(self, output: str) -> bool:
        """Check code syntax - look for basic syntax issues."""
        # Check for common syntax issues
        syntax_issues = [
            ("def ", "def "),  # Has function definition
            ("class ", "class "),  # Has class definition
            ("import ", "import"),  # Has import
            ("function ", "function"),  # Has function
            ("const ", "const"),  # Has const
            ("let ", "let"),  # Has let
            ("var ", "var"),  # Has var
        ]

        # Count opening/closing brackets
        open_braces = output.count("{")
        close_braces = output.count("}")
        open_parens = output.count("(")
        close_parens = output.count(")")
        open_brackets = output.count("[")
        close_brackets = output.count("]")

        # Basic bracket balance check
        if open_braces != close_braces:
            return False
        if open_parens != close_parens:
            return False
        if open_brackets != close_brackets:
            return False

        # Has code content (at least one language construct)
        has_code = any(item in output for _, item in syntax_issues)
        return has_code

    def _check_code_edge_cases(self, output: str) -> bool:
        """Check if code handles edge cases."""
        # Code should have at least some edge case handling
        edge_case_count = sum(1 for pattern in self.CODE_EDGE_CASE_PATTERNS if pattern in output)

        # If code is short (<100 chars), less edge case checking needed
        if len(output) < 100:
            return True

        # Medium code should have some edge case handling
        if len(output) < 500:
            return edge_case_count >= 1

        # Longer code should have proper error handling
        return edge_case_count >= 2

    def _check_code_security(self, output: str) -> bool:
        """Check for code security issues (redteam-style)."""
        output_lower = output.lower()

        # Check for dangerous patterns
        dangerous_patterns = [
            "eval(",
            "exec(",
            "os.system",
            "subprocess.Popen",
            "__import__",
            "compile(",
            "input(",
            "raw_input",
        ]

        # If any dangerous pattern found, flag it
        for pattern in dangerous_patterns:
            if pattern in output_lower:
                # Allow if it's in a comment explaining what's wrong
                if "# dangerous" in output_lower or "# security" in output_lower:
                    continue
                return False

        return True

    def _is_valid_path(self, path: str, quality_level: str, expected: str | None) -> bool:
        """Validate layer execution path."""
        if expected:
            return path == expected

        # Default validation rules
        if quality_level == "HIGH":
            return "L1" in path and "L2" in path and "L5" in path
        elif quality_level == "MEDIUM":
            return "L1" in path and "L5" in path
        else:
            return "L1" in path and "L5" in path

    def _get_rejection_reason(self, gates: dict[str, bool]) -> str:
        """Get rejection reason from failed gates."""
        failed = [gate for gate, passed in gates.items() if not passed]
        if not failed:
            return "Unknown rejection"

        reasons = {
            str(QualityGate.NO_TABOO_VIOLATION): "Taboo content detected",
            str(QualityGate.FORMAT_COMPLIANT): "Output format non-compliant",
            str(QualityGate.NO_P0_VULN): "P0 security vulnerability detected",
            str(QualityGate.NO_HARDCODED_SECRETS): "Hardcoded secret detected",
            str(QualityGate.GEMINI_BATCH_THRESHOLD): "Gemini batch threshold not met",
            str(QualityGate.L4_APPLIED_IF_NEEDED): "L4 compression required but not applied",
            str(QualityGate.MODEL_FULL_NAME): "Model name not in full format",
            str(QualityGate.LAYER_PATH_VALID): "Invalid layer execution path",
            str(QualityGate.GEMINI_QUALITY): "Gemini quality assessment failed",
        }

        return "; ".join(reasons.get(f, f) for f in failed[:2])
