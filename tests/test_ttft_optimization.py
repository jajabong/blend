"""Tests for TTFT Speculative Execution (L1+L2 Racing)."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest


class TestTTFTSpeculativeExecution:
    """Test TTFT optimization via L1+L2 concurrent racing."""

    def test_process_speculative_race_for_high_complexity(self) -> None:
        """HIGH complexity triggers parallel L1 draft + L2 strategy execution."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=9, tier="HIGH", task_type="general",
                breakdown={}, route_decision="HIGH",
            )
            mock_scorer_cls.return_value = mock_scorer

            # Strategy mock
            mock_strategy = MagicMock()
            mock_strategy.generate.return_value = MagicMock(
                output=MagicMock(
                    plan=["Step 1", "Step 2"],
                    quality_redlines=["No injection"],
                    boundary_cases=["Empty"],
                    model_hint="Opus",
                    estimated_tokens=30,
                )
            )
            mock_strategy_cls.return_value = mock_strategy

            # Executor mock - returns pre-draft for L1 task
            mock_executor = MagicMock()
            mock_executor.execute.return_value = MagicMock(
                raw_output="pre-draft content",
                model_used="minimax",
                tokens_used=50,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            from blend.core.orchestrator import BlendOrchestrator
            from blend.core.semantic_cache import CacheResult
            orchestrator = BlendOrchestrator()
            # Ensure cache miss
            orchestrator.cache.get = MagicMock(return_value=CacheResult(hit=False, reason="cache_miss"))

            result = orchestrator.process("Design a complex distributed system")

            # Verify concurrent execution happened (executor called for pre-draft)
            # Recipe model: HIGH complexity uses _execute_recipe instead of execute
            assert mock_executor._execute_recipe.call_count >= 1 or mock_executor.execute.call_count >= 1
            assert "L2" in result.layer_path

    def test_process_no_race_for_low_complexity(self) -> None:
        """LOW complexity skips L2 strategy and uses sequential execution."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=2, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_strategy = MagicMock()
            mock_strategy_cls.return_value = mock_strategy

            mock_executor = MagicMock()
            mock_executor.execute.return_value = MagicMock(
                raw_output="simple output",
                model_used="minimax",
                tokens_used=20,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            from blend.core.orchestrator import BlendOrchestrator
            from blend.core.semantic_cache import CacheResult
            orchestrator = BlendOrchestrator()
            orchestrator.cache.get = MagicMock(return_value=CacheResult(hit=False, reason="cache_miss"))

            result = orchestrator.process("What is 2+2?")

            # LOW complexity should NOT call strategy generator
            mock_strategy.generate.assert_not_called()
            assert "L2" not in result.layer_path


class TestCacheHitOptimization:
    """Test semantic cache hit bypasses model execution."""

    def test_cache_hit_returns_early(self) -> None:
        """Cache hit returns cached response without L1/L2/L3 execution."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=3, tier="LOW", task_type="code",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor_cls.return_value = mock_executor

            mock_strategy = MagicMock()
            mock_strategy_cls.return_value = mock_strategy

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            from blend.core.orchestrator import BlendOrchestrator
            from blend.core.semantic_cache import CacheResult
            orchestrator = BlendOrchestrator()
            # Set up cache hit
            orchestrator.cache.get = MagicMock(return_value=CacheResult(
                hit=True,
                response="Cached grep result: Found 5 matches",
                model_used="haiku",
                tokens_saved=200,
                reason="exact_hash_match",
            ))

            result = orchestrator.process("grep -r 'TODO' src/")

            # Cache hit should skip executor
            mock_executor.execute.assert_not_called()
            assert result.final_output == "Cached grep result: Found 5 matches"
            assert result.model_used == "haiku"
            assert "CACHE" in result.layer_path
            assert result.quality_gate_passed is True