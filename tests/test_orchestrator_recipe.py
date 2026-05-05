"""Tests for orchestrator integration with Recipe model."""

from unittest.mock import MagicMock, patch


class TestOrchestratorRecipeIntegration:
    """Orchestrator should use Recipe for multi-stage execution."""

    def test_process_uses_recipe_for_medium_complexity(self) -> None:
        """orchestrator.process() for complexity 4+ uses _execute_recipe."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        with patch.object(orch.scorer, "score") as mock_score, \
             patch.object(orch.cache, "get") as mock_cache, \
             patch.object(orch.executor, "_select_recipe") as mock_sel, \
             patch.object(orch.executor, "_execute_recipe") as mock_exec, \
             patch.object(orch.verifier, "verify") as mock_verify, \
             patch.object(orch.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(total=5, tier="MEDIUM", task_type="general")
            mock_cache.return_value = MagicMock(hit=False)
            mock_sel.return_value = MagicMock(stages=[])
            mock_exec.return_value = MagicMock(raw_output="Recipe result", model_used="sonnet",
                                               tokens_used=50, thought=None)
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            orch.process("Design a system")

            mock_sel.assert_called_once()
            mock_exec.assert_called_once()

    def test_process_passes_l2_strategy_hints_to_recipe(self) -> None:
        """HIGH complexity: L2 strategy (plan/redlines) passed as strategy_hints to recipe."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        mock_l2_output = MagicMock()
        mock_l2_output.output.plan = ["step1", "step2"]
        mock_l2_output.output.quality_redlines = ["no secrets"]

        with patch.object(orch.scorer, "score") as mock_score, \
             patch.object(orch.cache, "get") as mock_cache, \
             patch.object(orch.strategy, "generate") as mock_strategy, \
             patch.object(orch.executor, "_select_recipe") as mock_sel, \
             patch.object(orch.executor, "_execute_recipe") as mock_exec, \
             patch.object(orch.verifier, "verify") as mock_verify, \
             patch.object(orch.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(total=8, tier="HIGH", task_type="general")
            mock_cache.return_value = MagicMock(hit=False)
            mock_strategy.return_value = mock_l2_output
            mock_sel.return_value = MagicMock(stages=[])
            mock_exec.return_value = MagicMock(raw_output="Done", model_used="sonnet",
                                               tokens_used=100, thought=None)
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            orch.process("Design a distributed system")

            # L2 strategy was generated
            mock_strategy.assert_called_once()

            # Recipe was selected with strategy_hints containing L2 plan
            call_kwargs = mock_sel.call_args[1]
            assert call_kwargs["strategy_hints"] is not None
            assert "plan" in call_kwargs["strategy_hints"]
            assert call_kwargs["strategy_hints"]["plan"] == ["step1", "step2"]

    def test_low_complexity_skips_recipe_uses_routing(self) -> None:
        """LOW complexity (1-2): orchestrator uses normal execute, not Recipe."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        with patch.object(orch.scorer, "score") as mock_score, \
             patch.object(orch.cache, "get") as mock_cache, \
             patch.object(orch.executor, "execute") as mock_exec, \
             patch.object(orch.executor, "_select_recipe") as mock_sel, \
             patch.object(orch.executor, "_execute_recipe") as mock_exec_recipe, \
             patch.object(orch.verifier, "verify") as mock_verify, \
             patch.object(orch.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(total=2, tier="LOW", task_type="general")
            mock_cache.return_value = MagicMock(hit=False)
            mock_exec.return_value = MagicMock(raw_output="Quick answer", model_used="haiku",
                                               tokens_used=10, thought=None)
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            orch.process("Hi")

            # LOW goes through normal execute, not Recipe
            mock_exec.assert_called_once()
            mock_sel.assert_not_called()
            mock_exec_recipe.assert_not_called()

    def test_recipe_stages_replace_hardcoded_draft_refine(self) -> None:
        """Recipe model replaces hardcoded L1 draft + L3 refine pattern in orchestrator."""
        from blend.core.orchestrator import BlendOrchestrator

        orch = BlendOrchestrator()
        mock_l2_output = MagicMock()
        mock_l2_output.output.plan = ["step1", "step2"]
        mock_l2_output.output.quality_redlines = ["no eval"]
        mock_l2_output.output.model_hint = "Sonnet"

        with patch.object(orch.scorer, "score") as mock_score, \
             patch.object(orch.cache, "get") as mock_cache, \
             patch.object(orch.strategy, "generate") as mock_strategy, \
             patch.object(orch.executor, "_select_recipe") as mock_sel, \
             patch.object(orch.executor, "_execute_recipe") as mock_exec, \
             patch.object(orch.verifier, "verify") as mock_verify, \
             patch.object(orch.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(total=6, tier="HIGH", task_type="general")
            mock_cache.return_value = MagicMock(hit=False)
            mock_strategy.return_value = mock_l2_output

            recipe_instance = MagicMock()
            recipe_instance.stages = [
                MagicMock(model="haiku", role="draft"),
                MagicMock(model="sonnet", role="refine"),
                MagicMock(model="gemini", role="verify"),
            ]
            mock_sel.return_value = recipe_instance
            mock_exec.return_value = MagicMock(raw_output="Result", model_used="sonnet",
                                               tokens_used=80, thought=None)
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            orch.process("Complex task")

            # Recipe selected with L2 strategy hints (because tier is HIGH)
            mock_sel.assert_called_once()
            call_kwargs = mock_sel.call_args[1]
            assert call_kwargs["complexity"] == 6
            assert call_kwargs["task_type"] == "general"
            # HIGH tier → L2 generated → strategy_hints not None
            assert call_kwargs["strategy_hints"] is not None
            assert "plan" in call_kwargs["strategy_hints"]
            mock_exec.assert_called_once()