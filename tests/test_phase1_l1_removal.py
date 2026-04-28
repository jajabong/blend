"""Tests for Phase 1: Remove L1 compression + L4 threshold 200T."""

from unittest.mock import MagicMock


class TestL4Threshold1000:
    """L4 should trigger at 1000T (aligned with L5 Gate 6 threshold)."""

    def test_threshold_1000_triggers_at_1001(self) -> None:
        """Token count > 1000 should trigger compression."""
        from blend.core.compression import CompressionTrigger

        trigger = CompressionTrigger(threshold=1000)
        assert trigger.should_compress(1001) is True
        assert trigger.should_compress(1000) is False
        assert trigger.should_compress(999) is False

    def test_default_threshold_is_1000(self) -> None:
        """CompressionTrigger default threshold should be 1000."""
        from blend.core.compression import CompressionTrigger

        trigger = CompressionTrigger()
        assert trigger.threshold == 1000

    def test_agent_mode_still_skips(self) -> None:
        """agent_mode=True should still skip regardless of threshold."""
        from blend.core.compression import CompressionTrigger

        trigger = CompressionTrigger(threshold=1000)
        assert trigger.should_compress(10000, agent_mode=True) is False


class TestL1CompressionRemoved:
    """L1 compress_prompt should never be called from orchestrator."""

    def test_smart_compress_always_returns_false(self) -> None:
        """_smart_compress should always return False (L1 compression removed)."""
        from blend.core.orchestrator import BlendOrchestrator

        orchestrator = BlendOrchestrator()
        for prompt in ["hi", "A" * 200, "B" * 500, "C" * 2000]:
            should_compress, result = orchestrator._smart_compress(prompt, complexity=5)
            assert should_compress is False, f"Expected False for prompt len={len(prompt)}"
            assert result is None, f"Expected None for prompt len={len(prompt)}"

    def test_process_l1_compressed_always_false(self) -> None:
        """OrchestratorResult.l1_compressed should always be False."""
        from unittest.mock import MagicMock, patch

        with patch("blend.core.orchestrator.ComplexityScorer") as mock_cls, \
             patch("blend.core.orchestrator.Executor") as mock_exec_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel"), \
             patch("blend.core.orchestrator.QualityVerifier") as mock_v_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enf_cls:

            mock_cls.return_value.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_exec_cls.return_value.execute.return_value = MagicMock(
                raw_output="result", model_used="haiku", tokens_used=50,
            )
            mock_v_cls.return_value.verify.return_value = MagicMock(passed=True)
            mock_enf_cls.return_value.enforce.return_value = MagicMock(
                allowed=True, violations=[],
            )

            from blend.core.orchestrator import BlendOrchestrator
            orchestrator = BlendOrchestrator()
            result = orchestrator.process("Write me a detailed essay about AI")
            assert result.l1_compressed is False

    def test_process_messages_l1_compressed_always_false(self) -> None:
        """process_messages OrchestratorResult.l1_compressed always False."""
        from unittest.mock import MagicMock, patch

        with patch("blend.core.orchestrator.ComplexityScorer") as mock_cls, \
             patch("blend.core.orchestrator.Executor") as mock_exec_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel"), \
             patch("blend.core.orchestrator.QualityVerifier") as mock_v_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enf_cls:

            mock_cls.return_value.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_exec_cls.return_value.execute_messages.return_value = MagicMock(
                content="result", model_used="haiku", tokens_used=50,
                finish_reason="stop", tool_calls=None,
            )
            mock_v_cls.return_value.verify.return_value = MagicMock(passed=True)
            mock_enf_cls.return_value.enforce.return_value = MagicMock(
                allowed=True, violations=[],
            )

            from blend.core.orchestrator import BlendOrchestrator
            orchestrator = BlendOrchestrator()
            result = orchestrator.process_messages(
                [{"role": "user", "content": "Explain quantum computing"}]
            )
            assert result.l1_compressed is False

    def test_stream_l1_compressed_always_false(self) -> None:
        """stream _blend metadata l1_compressed always False."""
        from unittest.mock import MagicMock, patch

        with patch("blend.core.orchestrator.ComplexityScorer") as mock_cls, \
             patch("blend.core.orchestrator.Executor") as mock_exec_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel"):

            mock_cls.return_value.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_exec_cls.return_value.stream.return_value = iter(["result"])

            from blend.core.orchestrator import BlendOrchestrator
            orchestrator = BlendOrchestrator()
            chunks = list(orchestrator.stream("A" * 1000))
            for chunk in chunks:
                assert chunk["_blend"]["l1_compressed"] is False

    def test_stream_messages_l1_compressed_always_false(self) -> None:
        """stream_messages _blend metadata l1_compressed always False."""
        from unittest.mock import MagicMock, patch

        with patch("blend.core.orchestrator.ComplexityScorer") as mock_cls, \
             patch("blend.core.orchestrator.Executor") as mock_exec_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel"):

            mock_cls.return_value.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_exec_cls.return_value.stream_messages.return_value = iter([{"delta": {"content": "hi"}}])

            from blend.core.orchestrator import BlendOrchestrator
            orchestrator = BlendOrchestrator()
            chunks = list(
                orchestrator.stream_messages([{"role": "user", "content": "B" * 2000}])
            )
            for chunk in chunks:
                assert chunk["_blend"]["l1_compressed"] is False


class TestOrchestratorNoCompressImport:
    """orchestrator.py should not import compress_prompt."""

    def test_no_compress_prompt_import(self) -> None:
        """orchestrator module should not import compress_prompt."""

        import blend.core.orchestrator as orchestrator_module

        # Check source code doesn't import compress_prompt
        source = orchestrator_module.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "compress_prompt" not in content, (
                "orchestrator.py should not reference compress_prompt after L1 removal"
            )
            assert "from blend.utils.compress import" not in content, (
                "orchestrator.py should not import from blend.utils.compress"
            )


class TestL1OutputDataclass:
    """L1Output dataclass still needed for routing layer (ComplexityScorer)."""

    def test_l1output_exists_and_functional(self) -> None:
        """L1Output should still exist (routing/scoring preserved)."""
        from blend.core.layers import L1Output

        output = L1Output(
            compressed_prompt="unchanged",
            complexity_score=5,
            complexity_breakdown={"steps": 1, "domain": 2},
            route_decision="L3_HAIKU",
            l1_compressed=False,
            compression_ratio=0.0,
        )
        assert output.compressed_prompt == "unchanged"
        assert output.complexity_score == 5
        assert output.l1_compressed is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_orchestrator_without_compress():
    """Mock all orchestrator dependencies."""
    from unittest.mock import patch

    return patch.multiple(
        "blend.core.orchestrator",
        ComplexityScorer=MagicMock(),
        Executor=MagicMock(),
        StrategyGenerator=MagicMock(),
        ResourceModel=MagicMock(),
        QualityVerifier=MagicMock(),
        Enforcer=MagicMock(),
    )
