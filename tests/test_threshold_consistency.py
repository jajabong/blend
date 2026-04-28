"""Tests for threshold consistency between executor.py and budget.py."""

from blend.core.budget import ResourceModel


def test_executor_budget_matches_resource_model() -> None:
    """Executor._get_budget should match ResourceModel.get_budget values."""
    from blend.core.executor import Executor

    executor = Executor()
    resource_model = ResourceModel()

    # These should match
    expected = {
        "minimax": resource_model.get_budget("minimax"),
        "haiku": resource_model.get_budget("haiku"),
        "sonnet": resource_model.get_budget("sonnet"),
        "opus": resource_model.get_budget("opus"),
        "gemini": 200000,  # Gemini not in ResourceModel, hardcoded is OK
    }

    for model in ["minimax", "haiku", "sonnet", "opus"]:
        executor_budget = executor._get_budget(model)
        assert executor_budget == expected[model], (
            f"Executor budget for {model}: {executor_budget} != ResourceModel: {expected[model]}"
        )


def test_resource_model_has_required_models() -> None:
    """ResourceModel should have budgets for all models used in executor."""
    rm = ResourceModel()

    required_models = ["minimax", "haiku", "sonnet", "opus"]
    for model in required_models:
        budget = rm.get_budget(model)
        assert budget > 0, f"ResourceModel should have budget for {model}"
