"""Resource Model - Budget Tracking + Degradation Strategy."""

from dataclasses import dataclass
from enum import IntEnum


class BudgetStatus(IntEnum):
    """Budget status levels."""

    HEALTHY = 0
    WARNING = 1
    CRITICAL = 2

    @classmethod
    def from_usage_percent(
        cls,
        usage_percent: float,
        warning_threshold: float = 80,
    ) -> "BudgetStatus":
        """Determine status from usage percentage."""
        if usage_percent >= warning_threshold + 1:  # > 80 = critical
            return cls.CRITICAL
        elif usage_percent >= warning_threshold * 0.9:  # >= 72 = warning
            return cls.WARNING
        return cls.HEALTHY


@dataclass(frozen=True)
class ModelBudget:
    """Budget for a specific model."""

    name: str
    initial: int
    remaining: int
    usage_percent: float
    status: BudgetStatus


class BudgetTracker:
    """Tracks token budget for a model."""

    def __init__(self, initial_budget: int) -> None:
        """Initialize tracker with initial budget."""
        self._initial = initial_budget
        self._remaining = initial_budget

    @property
    def remaining(self) -> int:
        """Get remaining budget."""
        return self._remaining

    @property
    def usage_percent(self) -> float:
        """Get usage percentage."""
        if self._initial == 0:
            return 100.0
        return (1 - self._remaining / self._initial) * 100

    @property
    def status(self) -> BudgetStatus:
        """Get current budget status."""
        return BudgetStatus.from_usage_percent(self.usage_percent)

    def consume(self, tokens: int) -> None:
        """Consume tokens from budget."""
        self._remaining = max(0, self._remaining - tokens)

    def reset(self) -> None:
        """Reset budget to initial amount."""
        self._remaining = self._initial


class DegradationStrategy:
    """Determines model degradation when budget is low."""

    # Model hierarchy (from highest to lowest)
    HIERARCHY = ["opus", "sonnet", "haiku", "minimax"]

    # Usage threshold for degradation
    DEGRADATION_THRESHOLD = 90  # 90%

    def should_degrade(self, model: str, usage_percent: float) -> int:
        """Check if model should degrade.

        Returns:
            Degradation level (0=none, 1=one step, 2=two steps)
        """
        if usage_percent < self.DEGRADATION_THRESHOLD:
            return 0

        # Calculate degradation based on how over threshold
        excess = usage_percent - self.DEGRADATION_THRESHOLD
        if excess > 5:
            return 2  # Immediate degradation
        return 1

    def get_degraded_model(self, model: str, usage_percent: float) -> str:
        """Get degraded model for given model and usage.

        Args:
            model: Current model name
            usage_percent: Current usage percentage

        Returns:
            Degraded model name
        """
        degradation = self.should_degrade(model, usage_percent)
        model_lower = model.lower()

        try:
            current_level = self.HIERARCHY.index(model_lower)
        except ValueError:
            return model_lower

        # Move down hierarchy
        new_level = min(current_level + degradation, len(self.HIERARCHY) - 1)
        return self.HIERARCHY[new_level]


class ResourceModel:
    """Manages all model budgets and resources."""

    # Monthly budgets (tokens)
    BUDGETS = {
        "minimax": 100_000_000,  # Unlimited effectively
        "haiku": 1_000_000,
        "sonnet": 1_000_000,
        "opus": 500_000,
        "gemini": 200_000,
    }

    # Monthly costs (CNY)
    COSTS = {
        "minimax": 20,
        "haiku": 0,  # Bundled with Sonnet
        "sonnet": 0,  # Bundled
        "opus": 688,  # Monthly package (matches models.yaml)
        "gemini": 100,  # Estimated per 200k
    }

    def __init__(self) -> None:
        """Initialize resource model."""
        self._trackers: dict[str, BudgetTracker] = {}
        for name, budget in self.BUDGETS.items():
            self._trackers[name] = BudgetTracker(budget)

    def get_budget(self, model: str) -> int:
        """Get budget for model."""
        return self.BUDGETS.get(model.lower(), 0)

    def track_consumption(self, model: str, tokens: int) -> None:
        """Track token consumption for model."""
        tracker = self._trackers.get(model.lower())
        if tracker:
            tracker.consume(tokens)

    def get_status(self, model: str) -> ModelBudget:
        """Get current budget status for model."""
        tracker = self._trackers.get(model.lower())
        if not tracker:
            return ModelBudget(
                name=model,
                initial=0,
                remaining=0,
                usage_percent=100,
                status=BudgetStatus.CRITICAL,
            )

        return ModelBudget(
            name=model,
            initial=tracker._initial,
            remaining=tracker.remaining,
            usage_percent=tracker.usage_percent,
            status=tracker.status,
        )

    def get_remaining(self, model: str) -> int:
        """Get remaining tokens for model."""
        tracker = self._trackers.get(model.lower())
        if not tracker:
            return 0
        return tracker.remaining

    def should_degrade(self, model: str) -> bool:
        """Check if model should degrade."""
        tracker = self._trackers.get(model.lower())
        if not tracker:
            return False
        return tracker.status == BudgetStatus.CRITICAL

    def get_degraded_model(self, model: str) -> str:
        """Get degraded model for given model."""
        status = self.get_status(model)
        strategy = DegradationStrategy()
        return strategy.get_degraded_model(model, status.usage_percent)

    def estimate_monthly_cost(self) -> float:
        """Estimate monthly cost based on current usage."""
        total = sum(self.COSTS.values())

        # Add Gemini cost estimation (10% of requests)
        total += 100  # Estimate

        return total

    def reset_all(self) -> None:
        """Reset all budgets (monthly reset)."""
        for tracker in self._trackers.values():
            tracker.reset()
