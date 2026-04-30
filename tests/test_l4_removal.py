"""Tests for L4 removal — threshold fix and gate alignment.

Story: L4 阈值修复
Changes:
  - CompressionTrigger threshold: 200 → 1000
  - L5 Gate 6: output_tokens > 500 → output_tokens > 1000
  - CompressionTrigger in orchestrator: removed from init (unused)
  - orchestrator init no longer instantiates L4Compressor
  - L4 layer path never added (always l4_applied=False)
"""

from blend.core.compression import CompressionTrigger


class TestCompressionTriggerFixed:
    """CompressionTrigger threshold aligned with L5 Gate (1000 tokens)."""

    def test_trigger_above_1000(self) -> None:
        """Should trigger when tokens > 1000."""
        trigger = CompressionTrigger()
        assert trigger.should_compress(1001) is True

    def test_no_trigger_at_1000(self) -> None:
        """Should not trigger at exactly 1000 tokens."""
        trigger = CompressionTrigger()
        assert trigger.should_compress(1000) is False

    def test_no_trigger_below_1000(self) -> None:
        """Should not trigger below 1000 tokens."""
        trigger = CompressionTrigger()
        assert trigger.should_compress(999) is False

    def test_trigger_at_200_old_threshold(self) -> None:
        """OLD threshold was 200. Should NOT trigger at 200 with new threshold."""
        trigger = CompressionTrigger()
        assert trigger.should_compress(200) is False

    def test_threshold_configurable_still_works(self) -> None:
        """Threshold should still be configurable for special cases."""
        trigger = CompressionTrigger(threshold=500)
        assert trigger.should_compress(501) is True
        assert trigger.should_compress(500) is False


class TestL5Gate6Aligned:
    """L5 Gate 6 threshold aligned with CompressionTrigger (1000 tokens)."""

    def test_l5_gate6_no_fail_under_1000(self) -> None:
        """Gate 6 should pass when output < 1000T even if L4 not applied."""
        from blend.core.verifier import QualityVerifier

        verifier = QualityVerifier()
        # output_tokens=500, L4 not applied — Gate 6 should pass (not fail)
        result = verifier.verify(
            output="Short response" * 50,  # ~500 tokens
            quality_level="HIGH",
            layer_path="L1>L3>L5",
            output_tokens=500,
            l4_applied=False,
        )
        assert result.gates_checked["l4_applied_if_needed"] is True

    def test_l5_gate6_always_passes_l4_removed(self) -> None:
        """L4 has been removed - Gate 6 always passes regardless of output size."""
        from blend.core.verifier import QualityVerifier

        verifier = QualityVerifier()
        result = verifier.verify(
            output="Long response " * 300,  # ~1000+ tokens
            quality_level="HIGH",
            layer_path="L1>L3>L5",
            output_tokens=1100,
            l4_applied=False,
        )
        # Gate 6 always passes since L4 is removed
        assert result.gates_checked["l4_applied_if_needed"] is True

    def test_l5_gate6_pass_above_1000_with_l4(self) -> None:
        """Gate 6 should pass when output > 1000T and L4 was applied."""
        from blend.core.verifier import QualityVerifier

        verifier = QualityVerifier()
        result = verifier.verify(
            output="Compressed response " * 100,
            quality_level="HIGH",
            layer_path="L1>L3>L5>L4>L5",
            output_tokens=300,
            l4_applied=True,
        )
        # Should pass since L4 was applied
        assert result.gates_checked["l4_applied_if_needed"] is True

    def test_l5_gate6_pass_under_1000_without_l4(self) -> None:
        """Gate 6 should pass when output < 1000T even without L4."""
        from blend.core.verifier import QualityVerifier

        verifier = QualityVerifier()
        result = verifier.verify(
            output="Short.",
            quality_level="LOW",
            layer_path="L1>L3>L5",
            output_tokens=5,
            l4_applied=False,
        )
        assert result.gates_checked["l4_applied_if_needed"] is True


class TestOrchestratorNoL4:
    """Orchestrator should not add L4 to layer path (L4 compression removed)."""

    def test_orchestrator_init_no_l4_compressor(self) -> None:
        """Orchestrator should NOT instantiate L4Compressor."""
        from blend.core.orchestrator import BlendOrchestrator

        orchestrator = BlendOrchestrator()
        # L4Compressor removed from orchestrator
        assert not hasattr(orchestrator, "l4_compressor")

    def test_process_no_l4_layer_path(self) -> None:
        """process() should never add L4 to layer path."""
        from unittest.mock import MagicMock, patch

        from blend.core.orchestrator import BlendOrchestrator

        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=3, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.execute.return_value = MagicMock(
                raw_output="x", model_used="minimax", tokens_used=2000,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            orchestrator = BlendOrchestrator()
            result = orchestrator.process("A" * 400)

            # L4 should NEVER appear in layer path
            assert "L4" not in result.layer_path
            # l4_applied should always be False
            assert result.l4_applied is False

    def test_process_messages_no_l4(self) -> None:
        """process_messages() should never add L4 to layer path."""
        from unittest.mock import MagicMock, patch

        from blend.core.orchestrator import BlendOrchestrator

        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=9, tier="HIGH", task_type="general",
                breakdown={}, route_decision="HIGH",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.execute_messages.return_value = MagicMock(
                content="A" * 2000, model_used="sonnet", tokens_used=2000,
                finish_reason="stop", tool_calls=None,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            orchestrator = BlendOrchestrator()
            result = orchestrator.process_messages(
                [{"role": "user", "content": "Write a very long essay"}]
            )

            assert "L4" not in result.layer_path
            assert result.l4_applied is False


class TestOldL4TestsMarkedDeprecated:
    """Old test_l4.py tests that relied on threshold=200 should fail or skip."""

    def test_l4_default_threshold_no_longer_200(self) -> None:
        """Old default was 200. New default is 1000. Verify 200 doesn't trigger."""
        trigger = CompressionTrigger()
        # This was the old test: assert trigger.should_compress(201) is True
        # Now it should be False
        assert trigger.should_compress(201) is False
        assert trigger.should_compress(1001) is True

    def test_orchestrator_old_l4_tests_fail(self) -> None:
        """Old test: test_orchestrator_initialization checks l4_compressor exists."""
        from blend.core.orchestrator import BlendOrchestrator

        orchestrator = BlendOrchestrator()
        # Old test: assert orchestrator.l4_compressor is not None
        # New: l4_compressor removed
        assert not hasattr(orchestrator, "l4_compressor")
