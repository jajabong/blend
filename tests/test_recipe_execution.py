"""Tests for Recipe execution - Issue: Serial execution may overestimate draft value."""

import pytest
from unittest.mock import patch, MagicMock

from blend.core.executor import Executor, Recipe, RecipeStage


class TestRecipeSerialExecution:
    """Test that Recipe stages are executed serially (current behavior)."""

    def test_recipe_stages_are_sequential(self):
        """Recipe stages should be executed sequentially, not in parallel."""
        executor = Executor()

        # Create a recipe with 3 stages
        recipe = Recipe(stages=[
            RecipeStage(model="minimax", role="draft", complexity=1),
            RecipeStage(model="sonnet", role="refine", complexity=5),
            RecipeStage(model="gemini", role="verify", complexity=3),
        ])

        # Current behavior: sequential execution
        # This test documents the current (potentially suboptimal) behavior
        with patch.object(executor, '_call_model') as mock_call:
            mock_call.return_value = MagicMock(content="test output", usage={"completion_tokens": 100})

            result = executor._execute_recipe(recipe, "test prompt")

            # Verify call order
            calls = mock_call.call_args_list
            assert len(calls) == 3

            # First call: draft stage
            assert calls[0][1]['model'] == 'minimax'
            assert calls[0][1]['prompt'] == "Provide a detailed technical outline/draft for: test prompt"

            # Second call: refine stage
            assert calls[1][1]['model'] == 'sonnet'
            assert 'Existing Draft' in calls[1][1]['prompt']

    def test_draft_stage_value_questionable(self):
        """Draft stage value may be overestimated - draft is often not used effectively."""
        executor = Executor()

        # For HIGH complexity, draft is generated but its utility is unclear
        recipe = executor._select_recipe(complexity=7, task_type="general")

        # Verify draft is included
        stage_roles = [s.role for s in recipe.stages]
        assert "draft" in stage_roles

        # But the draft output is only used if refine stage exists
        # The verify stage doesn't use draft, just the refine output
        # This suggests draft value chain is broken


class TestRecipeParallelism:
    """Test if parallel execution is possible/desirable."""

    def test_independent_stages_could_be_parallel(self):
        """For HIGH complexity, draft and strategy could run in parallel."""
        # Strategy (L2) and Draft (L1) are independent
        # They could run in parallel before the refine stage

        # This is a design suggestion, not a bug
        pass