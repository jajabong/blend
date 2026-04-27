"""Tests for story-008: Resource Model - Budget Tracking + Degradation Strategy."""

from blend.core.budget import (
    BudgetStatus,
    BudgetTracker,
    DegradationStrategy,
    ResourceModel,
)


class TestBudgetTracker:
    """Test budget tracking."""

    def test_initial_budget_full(self) -> None:
        """New tracker should have full budget."""
        tracker = BudgetTracker(initial_budget=100000)
        assert tracker.remaining == 100000
        assert tracker.usage_percent == 0

    def test_consume_reduces_budget(self) -> None:
        """Consuming tokens should reduce budget."""
        tracker = BudgetTracker(initial_budget=1000)
        tracker.consume(100)
        assert tracker.remaining == 900

    def test_usage_percent_calculation(self) -> None:
        """Usage percent should be calculated correctly."""
        tracker = BudgetTracker(initial_budget=1000)
        tracker.consume(500)
        assert tracker.usage_percent == 50

    def test_reset_restores_budget(self) -> None:
        """Reset should restore full budget."""
        tracker = BudgetTracker(initial_budget=1000)
        tracker.consume(500)
        tracker.reset()
        assert tracker.remaining == 1000


class TestBudgetStatus:
    """Test budget status levels."""

    def test_healthy_under_warning(self) -> None:
        """Budget should be healthy when under warning threshold."""
        status = BudgetStatus.from_usage_percent(50, warning_threshold=80)
        assert status == BudgetStatus.HEALTHY

    def test_warning_at_threshold(self) -> None:
        """Budget should be warning at threshold."""
        status = BudgetStatus.from_usage_percent(80, warning_threshold=80)
        assert status == BudgetStatus.WARNING

    def test_critical_above_threshold(self) -> None:
        """Budget should be critical above threshold."""
        status = BudgetStatus.from_usage_percent(95, warning_threshold=80)
        assert status == BudgetStatus.CRITICAL


class TestDegradationStrategy:
    """Test degradation strategy."""

    def test_no_degradation_healthy(self) -> None:
        """No degradation when budget is healthy."""
        strategy = DegradationStrategy()
        degradation = strategy.should_degrade(
            model="sonnet",
            usage_percent=50,
        )
        assert degradation == 0

    def test_degradation_critical(self) -> None:
        """Should degrade when budget is critical."""
        strategy = DegradationStrategy()
        degradation = strategy.should_degrade(
            model="sonnet",
            usage_percent=95,
        )
        assert degradation > 0

    def test_haiku_degrades_to_minimax(self) -> None:
        """Haiku should degrade to Minimax at critical."""
        strategy = DegradationStrategy()
        result = strategy.get_degraded_model("haiku", 95)
        assert result == "minimax"

    def test_sonnet_degrades_to_haiku(self) -> None:
        """Sonnet should degrade to Haiku at critical."""
        strategy = DegradationStrategy()
        result = strategy.get_degraded_model("sonnet", 95)
        assert result == "haiku"

    def test_opus_degrades_to_sonnet(self) -> None:
        """Opus should degrade to Sonnet at critical."""
        strategy = DegradationStrategy()
        result = strategy.get_degraded_model("opus", 95)
        assert result == "sonnet"

    def test_minimax_no_degradation(self) -> None:
        """Minimax should not degrade further."""
        strategy = DegradationStrategy()
        result = strategy.get_degraded_model("minimax", 95)
        assert result == "minimax"


class TestResourceModel:
    """Test resource model."""

    def test_model_budgets_defined(self) -> None:
        """All models should have budgets defined."""
        model = ResourceModel()
        assert model.get_budget("minimax") > 0
        assert model.get_budget("haiku") > 0
        assert model.get_budget("sonnet") > 0
        assert model.get_budget("opus") > 0

    def test_track_consumption(self) -> None:
        """Should track token consumption."""
        model = ResourceModel()
        model.track_consumption("haiku", 100)
        status = model.get_status("haiku")
        assert status.usage_percent > 0

    def test_check_degradation_threshold(self) -> None:
        """Should detect when degradation is needed."""
        model = ResourceModel()
        # Consume 95% of budget
        budget = model.get_budget("sonnet")
        model.track_consumption("sonnet", int(budget * 0.95))

        should_degrade = model.should_degrade("sonnet")
        assert should_degrade is True

    def test_total_monthly_cost_estimate(self) -> None:
        """Should estimate total monthly cost."""
        model = ResourceModel()
        cost = model.estimate_monthly_cost()
        assert cost > 0
        assert cost <= 1000  # Target: <= 808 CNY
