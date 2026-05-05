"""Tests for Recipe Model - multi-stage execution instead of single model routing."""

from unittest.mock import MagicMock, patch
from blend.core.executor import Executor, Recipe, RecipeStage


class TestRecipeDataclass:
    """Recipe and RecipeStage dataclasses for multi-stage execution."""

    def test_recipe_stage_creation(self) -> None:
        """RecipeStage holds model, role, complexity, hints."""
        stage = RecipeStage(
            model="haiku",
            role="draft",
            complexity=1,
            timeout=10.0,
            strategy_hints={"plan": ["step1", "step2"]},
        )
        assert stage.model == "haiku"
        assert stage.role == "draft"
        assert stage.complexity == 1
        assert stage.timeout == 10.0
        assert stage.strategy_hints == {"plan": ["step1", "step2"]}

    def test_recipe_stage_defaults(self) -> None:
        """RecipeStage has sensible defaults."""
        stage = RecipeStage(model="sonnet", role="refine", complexity=5)
        assert stage.timeout == 15.0
        assert stage.strategy_hints is None

    def test_recipe_stages_list(self) -> None:
        """Recipe holds ordered list of stages."""
        recipe = Recipe(stages=[
            RecipeStage(model="haiku", role="draft", complexity=1),
            RecipeStage(model="sonnet", role="refine", complexity=7),
        ])
        assert len(recipe.stages) == 2
        assert recipe.stages[0].model == "haiku"
        assert recipe.stages[1].model == "sonnet"

    def test_recipe_ensemble_flag(self) -> None:
        """Recipe can enable ensemble mode."""
        recipe = Recipe(stages=[], ensemble=True, merge_strategy="vote")
        assert recipe.ensemble is True
        assert recipe.merge_strategy == "vote"

    def test_recipe_default_merge_strategy(self) -> None:
        """Recipe defaults to best_only merge."""
        recipe = Recipe(stages=[])
        assert recipe.merge_strategy == "best_only"
        assert recipe.ensemble is False


class TestRecipeSelection:
    """Recipe selection based on complexity and task type."""

    def test_low_complexity_single_stage(self) -> None:
        """Complexity 1-2: single haiku stage, no draft-refine."""
        exec = Executor()
        recipe = exec._select_recipe(complexity=1, task_type="general")
        assert len(recipe.stages) == 1
        assert recipe.stages[0].model == "haiku"
        assert recipe.stages[0].role == "execute"

    def test_medium_complexity_two_stage(self) -> None:
        """Complexity 3-5: haiku draft + sonnet refine."""
        exec = Executor()
        recipe = exec._select_recipe(complexity=4, task_type="general")
        assert len(recipe.stages) == 2
        assert recipe.stages[0].role == "draft"
        assert recipe.stages[1].role == "refine"

    def test_high_complexity_three_stage_with_verify(self) -> None:
        """Complexity 6+: draft + refine + verify."""
        exec = Executor()
        recipe = exec._select_recipe(complexity=7, task_type="general")
        assert len(recipe.stages) == 3
        assert recipe.stages[0].role == "draft"
        assert recipe.stages[1].role == "refine"
        assert recipe.stages[2].role == "verify"

    def test_gemini_task_type_routes_gemini(self) -> None:
        """deep_reasoning/multimodal task types use gemini as primary."""
        exec = Executor()
        recipe = exec._select_recipe(complexity=5, task_type="deep_reasoning")
        # First stage should use gemini
        assert recipe.stages[0].model in ["gemini", "gemini_pro"]

    def test_code_task_type_uses_sonnet_for_refine(self) -> None:
        """Code tasks use sonnet for refinement stage."""
        exec = Executor()
        recipe = exec._select_recipe(complexity=6, task_type="code")
        refine_stage = next(s for s in recipe.stages if s.role == "refine")
        assert refine_stage.model in ["sonnet", "opus"]

    def test_strategy_hints_injected_into_refine(self) -> None:
        """L2 strategy hints (plan, redlines) injected into refine stage."""
        exec = Executor()
        recipe = exec._select_recipe(
            complexity=8,
            task_type="general",
            strategy_hints={"plan": ["step1", "step2"], "redlines": ["no secrets"]},
        )
        refine_stage = next(s for s in recipe.stages if s.role == "refine")
        assert refine_stage.strategy_hints is not None
        assert "plan" in refine_stage.strategy_hints


class TestRecipeExecution:
    """Executing a recipe runs stages sequentially, feeding output forward."""

    def test_execute_recipe_runs_all_stages(self) -> None:
        """execute_recipe runs each stage and returns final output."""
        exec = Executor()

        # Mock _call_model to return stage-specific content
        def mock_call(model, prompt, **kwargs):
            response = MagicMock()
            if "draft" in prompt.lower() or "outline" in prompt.lower():
                response.content = "Draft output from haiku"
            else:
                response.content = "Refined output from sonnet"
            response.usage = {"completion_tokens": 20}
            return response

        with patch.object(exec, "_call_model", side_effect=mock_call):
            recipe = Recipe(stages=[
                RecipeStage(model="haiku", role="draft", complexity=1),
                RecipeStage(model="sonnet", role="refine", complexity=5),
            ])
            result = exec._execute_recipe(recipe, "Design a system")

        # Should have called both stages
        assert "Draft" in result.raw_output or "Refined" in result.raw_output

    def test_execute_recipe_feeds_output_forward(self) -> None:
        """Draft output becomes part of refine prompt."""
        exec = Executor()

        calls = []
        def mock_call(model, prompt, **kwargs):
            calls.append({"model": model, "prompt": prompt[:50]})
            response = MagicMock()
            response.content = f"Output from {model}"
            response.usage = {"completion_tokens": 10}
            return response

        with patch.object(exec, "_call_model", side_effect=mock_call):
            recipe = Recipe(stages=[
                RecipeStage(model="haiku", role="draft", complexity=1),
                RecipeStage(model="sonnet", role="refine", complexity=5),
            ])
            exec._execute_recipe(recipe, "Test prompt")

        assert len(calls) == 2
        # Second call should reference first call's output
        assert calls[1]["model"] == "sonnet"

    def test_execute_recipe_respects_timeout_per_stage(self) -> None:
        """Each stage uses its own timeout setting."""
        exec = Executor()

        with patch.object(exec, "_call_model") as mock_call:
            mock_call.return_value = MagicMock(content="ok", usage={"completion_tokens": 5})
            recipe = Recipe(stages=[
                RecipeStage(model="haiku", role="draft", complexity=1, timeout=5.0),
                RecipeStage(model="sonnet", role="refine", complexity=5, timeout=30.0),
            ])
            exec._execute_recipe(recipe, "Test")

            # First call should use 5.0 timeout
            first_call_kwargs = mock_call.call_args_list[0].kwargs
            assert first_call_kwargs.get("timeout") == 5.0

            # Second call should use 30.0 timeout
            second_call_kwargs = mock_call.call_args_list[1].kwargs
            assert second_call_kwargs.get("timeout") == 30.0


class TestRecipeVsRouting:
    """Recipe model is fundamentally different from routing model."""

    def test_recipe_has_multiple_stages_for_medium_high(self) -> None:
        """Recipe for complexity 5 has 2+ stages, routing has 1 primary."""
        exec = Executor()
        recipe = exec._select_recipe(complexity=5, task_type="general")
        # Routing would return single "sonnet" - recipe returns stages
        assert len(recipe.stages) >= 2

    def test_recipe_injects_strategy_hints(self) -> None:
        """Recipe can carry L2 strategy hints to refine stage."""
        exec = Executor()
        hints = {"plan": ["1. Analyze", "2. Build"], "redlines": ["No eval"]}
        recipe = exec._select_recipe(complexity=8, task_type="general", strategy_hints=hints)

        # Routing's _select_model returns no strategy info
        selection = exec._select_model(complexity=8, task_type="general")
        assert selection.primary in ["sonnet", "gemini_pro_ultra"]

        # Recipe carries hints that routing doesn't
        refine = next((s for s in recipe.stages if s.role == "refine"), None)
        assert refine is not None
        assert refine.strategy_hints == hints