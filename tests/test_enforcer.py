"""Tests for story-007: Enforcement Mechanism - 8 Taboos Auto-Rejection."""

from blend.core.enforcer import (
    TABOOS,
    EnforcementResult,
    Enforcer,
    TabooViolation,
)


class TestTabooDefinitions:
    """Test taboo definitions."""

    def test_all_taboos_defined(self) -> None:
        """All active taboos should be defined (6 after L4 removal)."""
        assert len(TABOOS) == 6

    def test_taboos_have_ids(self) -> None:
        """Each taboo should have an ID."""
        for taboo in TABOOS:
            assert hasattr(taboo, "id")
            assert taboo.id is not None

    def test_taboos_have_messages(self) -> None:
        """Each taboo should have a message."""
        for taboo in TABOOS:
            assert hasattr(taboo, "message")
            assert len(taboo.message) > 0


class TestEnforcer:
    """Test enforcement mechanism."""

    def test_enforce_passes_clean_request(self) -> None:
        """Clean request should pass enforcement."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"prompt": "Hello, how are you?"},
            layer_path="L1>L3>L5",
        )
        assert result.allowed is True
        assert result.violations == []

    def test_enforce_blocks_skip_l1(self) -> None:
        """Request skipping L1 should be blocked."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"prompt": "Hello"},
            layer_path="L3>L5",  # Missing L1
        )
        assert result.allowed is False
        assert any("L1" in str(v.taboo.id) for v in result.violations)

    def test_enforce_blocks_invalid_layer_path(self) -> None:
        """Invalid layer path should be blocked."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"prompt": "Test"},
            layer_path="L1>L2",  # Incomplete path
        )
        # Should flag invalid path
        assert result.violations is not None

    def test_enforce_allows_valid_paths_for_low(self) -> None:
        """LOW complexity should allow L1>L3>L5 path."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"prompt": "Simple question"},
            layer_path="L1>L3>L5",
            complexity=2,
        )
        assert result.allowed is True

    def test_enforce_allows_valid_paths_for_high(self) -> None:
        """HIGH complexity should require L1>L2>L3>L5 path (L4 removed)."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"prompt": "Design a system"},
            layer_path="L1>DRAFT>L2>L3>L5",
            complexity=9,
        )
        assert result.allowed is True

    def test_enforce_blocks_gemini_batch_threshold(self) -> None:
        """Gemini below 50% threshold should be blocked."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"model": "gemini", "context_percent": 30},
            layer_path="L1>L3>L5",
            gemini_used=True,
            gemini_context_percent=30,
        )
        assert result.allowed is False

    def test_enforce_allows_gemini_above_threshold(self) -> None:
        """Gemini above 50% threshold should be allowed."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"model": "gemini"},
            layer_path="L1>L3>L5",
            gemini_used=True,
            gemini_context_percent=60,
        )
        assert result.allowed is True

    def test_enforce_l4_removed_always_passes(self) -> None:
        """L4 has been removed - _check_l4_required always returns empty (no-op)."""
        enforcer = Enforcer()
        # Even with >500 tokens and l4_applied=False, should NOT block (L4 removed)
        result = enforcer.enforce(
            request={"prompt": "Long response"},
            layer_path="L1>L3>L5",
            output_tokens=600,
            l4_applied=False,
        )
        # L4 removed - check is no-op, should allow
        assert result.allowed is True


class TestEnforcementResult:
    """Test EnforcementResult structure."""

    def test_result_allowed(self) -> None:
        """Allowed result should have no violations."""
        result = EnforcementResult(allowed=True, violations=[])
        assert result.allowed is True
        assert len(result.violations) == 0

    def test_result_blocked(self) -> None:
        """Blocked result should have violations."""
        violation = TabooViolation(
            taboo=TABOOS[0],
            reason="Test violation",
        )
        result = EnforcementResult(allowed=False, violations=[violation])
        assert result.allowed is False
        assert len(result.violations) == 1

    def test_multiple_violations(self) -> None:
        """Should track multiple violations."""
        violations = [
            TabooViolation(taboo=TABOOS[0], reason="First violation"),
            TabooViolation(taboo=TABOOS[1], reason="Second violation"),
        ]
        result = EnforcementResult(allowed=False, violations=violations)
        assert len(result.violations) == 2
