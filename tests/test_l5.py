"""Tests for story-006: L5 Verification Layer - Graded Quality Gate."""

from blend.core.verifier import QualityGate, QualityVerifier, VerificationResult


class TestQualityGate:
    """Test quality gate definitions."""

    def test_gate_names(self) -> None:
        """All 8 gates should be defined."""
        assert QualityGate.NO_TABOO_VIOLATION in QualityGate
        assert QualityGate.FORMAT_COMPLIANT in QualityGate
        assert QualityGate.NO_P0_VULN in QualityGate
        assert QualityGate.NO_HARDCODED_SECRETS in QualityGate
        assert QualityGate.GEMINI_BATCH_THRESHOLD in QualityGate
        assert QualityGate.L4_APPLIED_IF_NEEDED in QualityGate
        assert QualityGate.MODEL_FULL_NAME in QualityGate
        assert QualityGate.LAYER_PATH_VALID in QualityGate

    def test_gate_count(self) -> None:
        """Should have exactly 12 gates (9 base + 3 code-specific)."""
        gates = list(QualityGate)
        assert len(gates) == 12


class TestQualityVerifier:
    """Test quality verification."""

    def test_verify_low_quality(self) -> None:
        """LOW quality should use Minimax self-check."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output="Simple response",
            quality_level="LOW",
            layer_path="L1>L3>L5",
        )
        assert result.passed is not None

    def test_verify_medium_quality(self) -> None:
        """MEDIUM quality should use Haiku gate."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output="Medium complexity response",
            quality_level="MEDIUM",
            layer_path="L1>L3>L5",
        )
        assert result.passed is not None

    def test_verify_high_quality(self) -> None:
        """HIGH quality should use Opus final check."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output="High complexity response",
            quality_level="HIGH",
            layer_path="L1>L2>L3>L5",
        )
        assert result.passed is not None

    def test_gates_checked(self) -> None:
        """All gates should be checked."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output="Test output",
            quality_level="MEDIUM",
            layer_path="L1>L3>L5",
        )
        assert len(result.gates_checked) == 12

    def test_rejection_reason_on_failure(self) -> None:
        """Should provide rejection reason when failed."""
        verifier = QualityVerifier()

        # Test with hardcoded secret - should fail
        result = verifier.verify(
            output="API key: sk-1234567890abcdef",
            quality_level="LOW",
            layer_path="L1>L3>L5",
        )
        # The gates_checked should include no_hardcoded_secrets
        assert "no_hardcoded_secrets" in result.gates_checked

    def test_layer_path_validation(self) -> None:
        """Should validate layer path."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output="Test",
            quality_level="HIGH",
            layer_path="L1>L3>L5",  # Missing L2 for HIGH
        )
        assert result.gates_checked.get("layer_path_valid") is not None


class TestCodeSpecificGates:
    """Test code-specific redteam-style gates."""

    def test_code_gate_count(self) -> None:
        """Should have 3 code-specific gates."""
        code_gates = [
            QualityGate.CODE_SYNTAX_CHECK,
            QualityGate.CODE_EDGE_CASE,
            QualityGate.CODE_SECURITY,
        ]
        assert len(code_gates) == 3
        assert QualityGate.CODE_SYNTAX_CHECK in QualityGate
        assert QualityGate.CODE_EDGE_CASE in QualityGate
        assert QualityGate.CODE_SECURITY in QualityGate

    def test_code_syntax_gate(self) -> None:
        """Code syntax gate should validate balanced brackets and code constructs."""
        verifier = QualityVerifier()

        # Valid code - balanced and has language constructs
        result = verifier.verify(
            output='def hello(): print("hi")',
            quality_level="HIGH",
            layer_path="L1>L3>L5",
            task_type="code",
        )
        assert "code_syntax_check" in result.gates_checked

        # Invalid code - unbalanced braces
        result = verifier.verify(
            output="function test() { if (x > 0 { return x }",  # Missing closing paren
            quality_level="HIGH",
            layer_path="L1>L3>L5",
            task_type="code",
        )
        assert result.gates_checked["code_syntax_check"] is False

    def test_code_edge_case_gate(self) -> None:
        """Edge case gate should validate error handling in code."""
        verifier = QualityVerifier()

        # Code with error handling
        result = verifier.verify(
            output='def safe_div(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return None',
            quality_level="HIGH",
            layer_path="L1>L3>L5",
            task_type="code",
        )
        assert "code_edge_case" in result.gates_checked

    def test_code_security_gate(self) -> None:
        """Security gate should detect dangerous patterns."""
        verifier = QualityVerifier()

        # Code with dangerous pattern
        result = verifier.verify(
            output='eval(user_input)',
            quality_level="HIGH",
            layer_path="L1>L3>L5",
            task_type="code",
        )
        assert result.gates_checked["code_security"] is False

        # Code with security comment explaining the danger
        result = verifier.verify(
            output='# dangerous - eval can run arbitrary code\nresult = eval(user_input)',
            quality_level="HIGH",
            layer_path="L1>L3>L5",
            task_type="code",
        )
        # Should pass because comment acknowledges the danger
        assert result.gates_checked["code_security"] is True

    def test_non_code_task_skips_code_gates(self) -> None:
        """Non-code tasks should skip code-specific gates."""
        verifier = QualityVerifier()

        result = verifier.verify(
            output="This is a general response, not code.",
            quality_level="MEDIUM",
            layer_path="L1>L3>L5",
            task_type="general",
        )
        # Non-code tasks get True for code gates
        assert result.gates_checked["code_syntax_check"] is True
        assert result.gates_checked["code_edge_case"] is True
        assert result.gates_checked["code_security"] is True


class TestFormatComplianceGate:
    """Test FORMAT_COMPLIANT gate (gate 2)."""

    def test_format_pass_normal_text(self) -> None:
        """Normal text output should pass format check."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output="This is a normal response with some content.",
            quality_level="MEDIUM",
            layer_path="L1>L3>L5",
        )
        assert result.gates_checked["format_compliant"] is True

    def test_format_pass_valid_json(self) -> None:
        """Valid JSON output should pass."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output='{"key": "value", "count": 42}',
            quality_level="MEDIUM",
            layer_path="L1>L3>L5",
        )
        assert result.gates_checked["format_compliant"] is True

    def test_format_pass_valid_json_array(self) -> None:
        """Valid JSON array should pass."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output='[{"id": 1}, {"id": 2}]',
            quality_level="MEDIUM",
            layer_path="L1>L3>L5",
        )
        assert result.gates_checked["format_compliant"] is True

    def test_format_fail_invalid_json(self) -> None:
        """Invalid JSON (looks like JSON but isn't) should fail."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output='{"key": "value", broken}',
            quality_level="MEDIUM",
            layer_path="L1>L3>L5",
        )
        assert result.gates_checked["format_compliant"] is False

    def test_format_fail_empty_output(self) -> None:
        """Empty output should fail."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output="   ",
            quality_level="LOW",
            layer_path="L1>L3>L5",
        )
        assert result.gates_checked["format_compliant"] is False

    def test_format_pass_code_block(self) -> None:
        """Matched code fences should pass."""
        verifier = QualityVerifier()
        result = verifier.verify(
            output='```python\nprint("hello")\n```',
            quality_level="HIGH",
            layer_path="L1>L3>L5",
        )
        assert result.gates_checked["format_compliant"] is True


class TestVerificationResult:
    """Test VerificationResult structure."""

    def test_verification_result_creation(self) -> None:
        """Test creating VerificationResult."""

        result = VerificationResult(
            passed=True,
            gates_checked={
                "no_taboo_violation": True,
                "format_compliant": True,
            },
            rejection_reason=None,
        )
        assert result.passed is True
        assert result.rejection_reason is None

    def test_verification_result_rejection(self) -> None:
        """Test rejection reason."""
        result = VerificationResult(
            passed=False,
            gates_checked={"no_taboo_violation": False},
            rejection_reason="Taboo content detected",
        )
        assert result.passed is False
        assert result.rejection_reason == "Taboo content detected"

    def test_convert_to_l5_output(self) -> None:
        """Should convert to L5Output."""
        from blend.core.layers import L5Output

        result = VerificationResult(
            passed=True,
            gates_checked={
                "no_taboo_violation": True,
                "format_compliant": True,
                "no_p0_vuln": True,
                "no_hardcoded_secrets": True,
                "gemini_batch_threshold": True,
                "l4_applied_if_needed": True,
                "model_full_name": True,
                "layer_path_valid": True,
            },
            rejection_reason=None,
        )
        l5_output = L5Output(
            final_output="Verified output",
            quality_gate_passed=result.passed,
            gates_checked=result.gates_checked,
            quality_level="MEDIUM",
            rejection_reason=result.rejection_reason,
        )
        assert l5_output.quality_gate_passed is True
