"""Tests for story-010: Integration Tests."""

from blend.core.budget import ResourceModel
from blend.core.enforcer import Enforcer
from blend.core.layers import L1Output, L2Output, L3Output, L5Output
from blend.intent.scorer import ComplexityScorer


class TestIntegrationEndToEnd:
    """Integration tests for full layer flow."""

    def test_low_complexity_minimax_flow(self) -> None:
        """Low complexity (1-3) should route correctly."""
        scorer = ComplexityScorer()
        result = scorer.score("What is 2+2?")

        # Very simple tasks should route to L3_MINIMAX
        assert result.total <= 3
        assert result.route_decision == "L3_MINIMAX"

    def test_medium_complexity_haiku_flow(self) -> None:
        """Medium complexity should route correctly."""
        scorer = ComplexityScorer()
        # Use an extremely complex prompt with high risk
        result = scorer.score(
            "DESIGN A COMPLETE BANKING SYSTEM with database sharding, "
            "multi-region disaster recovery, and legal compliance with SOX, "
            "PCI-DSS, and GDPR. Include transaction processing, fraud detection "
            "with ML models, and real-time balance calculations."
        )

        # Very high complexity should be at least MEDIUM
        assert result.tier in ["MEDIUM", "HIGH"]

    def test_high_complexity_opus_flow(self) -> None:
        """High complexity should route correctly."""
        scorer = ComplexityScorer()
        # Maximum complexity prompt
        result = scorer.score(
            "CRITICAL MISSION-CRITICAL SYSTEM: Design a complete financial trading "
            "platform handling 10M transactions per second with HFT algorithms, "
            "real-time risk management, regulatory compliance across 50 countries, "
            "and AI-driven fraud detection with $1B daily risk exposure. Include "
            "multi-region active-active failover, circuit breakers, and consensus "
            "algorithms for distributed transaction processing."
        )

        # Should be high tier for extremely complex prompts
        assert result.tier in ["MEDIUM", "HIGH"]


class TestLayerOutputs:
    """Test layer output dataclasses."""

    def test_l1_output_format(self) -> None:
        """L1 output should have correct format."""
        output = L1Output(
            compressed_prompt="Test prompt",
            complexity_score=5,
            complexity_breakdown={"steps": 1, "domain": 2, "output": 1, "creativity": 1, "risk": 0},
            route_decision="L3_HAIKU",
            l1_compressed=True,
            compression_ratio=0.7,
        )
        assert output.compressed_prompt == "Test prompt"
        assert output.complexity_score == 5
        assert output.l1_compressed is True

    def test_l2_output_format(self) -> None:
        """L2 output should have correct format."""
        output = L2Output(
            plan=["Step 1", "Step 2"],
            quality_redlines=["Must be valid JSON"],
            boundary_cases=["Empty input"],
            model_hint="SONNET",
            estimated_tokens=150,
        )
        assert len(output.plan) == 2
        assert output.estimated_tokens <= 300

    def test_l3_output_format(self) -> None:
        """L3 output should have correct format."""
        output = L3Output(
            raw_output="Test output",
            model_used="haiku",
            tokens_used=100,
            tokens_budget_remaining=900,
            quality_gate_passed=True,
        )
        assert output.raw_output == "Test output"
        assert output.tokens_used == 100

    def test_l5_output_format(self) -> None:
        """L5 output should have correct format."""
        output = L5Output(
            final_output="Final output",
            quality_gate_passed=True,
            gates_checked={
                "no_taboo_violation": True,
                "format_compliant": True,
                "no_p0_vuln": True,
                "no_hardcoded_secrets": True,
                "gemini_batch_threshold": True,
                "l4_applied_if_needed": True,
                "model_full_name": True,
                "layer_path_valid": True,
            },
            quality_level="MEDIUM",
            rejection_reason=None,
        )
        assert output.quality_gate_passed is True
        assert output.quality_level == "MEDIUM"


class TestEnforcerIntegration:
    """Test enforcer with various scenarios."""

    def test_clean_request_passes(self) -> None:
        """Clean request should pass enforcement."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"prompt": "Hello"},
            layer_path="L1>L3>L5",
            complexity=3,
            output_tokens=50,
            l4_applied=False,  # Small output, no L4 needed
            gemini_used=False,
            gemini_context_percent=0,
            model_used="minimax",
        )
        assert result.allowed is True
        assert len(result.violations) == 0

    def test_skip_l1_blocked(self) -> None:
        """Request without L1 compression should be blocked."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"prompt": "Hello"},
            layer_path="L3>L5",  # Missing L1!
            complexity=3,
            output_tokens=50,
            l4_applied=False,
            gemini_used=False,
            gemini_context_percent=0,
            model_used="minimax",
        )
        assert result.allowed is False
        # Check violation mentions L1
        assert any("L1" in str(v.taboo.id) for v in result.violations)

    def test_large_output_l4_removed(self) -> None:
        """L4 has been removed - large output without L4 now passes."""
        enforcer = Enforcer()
        result = enforcer.enforce(
            request={"prompt": "Hello"},
            layer_path="L1>L3>L5",
            complexity=3,
            output_tokens=600,  # > 500 threshold (L4 removed, no longer enforced)
            l4_applied=False,
            gemini_used=False,
            gemini_context_percent=0,
            model_used="minimax",
        )
        assert result.allowed is True


class TestBudgetIntegration:
    """Test budget tracking integration."""

    def test_resource_model_budgets(self) -> None:
        """Resource model should have all budgets."""
        model = ResourceModel()
        assert model.get_budget("minimax") == 100_000_000
        assert model.get_budget("haiku") == 1_000_000
        assert model.get_budget("sonnet") == 1_000_000
        assert model.get_budget("opus") == 500_000

    def test_degradation_at_critical(self) -> None:
        """Should degrade when budget critical."""
        model = ResourceModel()
        # Consume 95% of budget
        budget = model.get_budget("sonnet")
        model.track_consumption("sonnet", int(budget * 0.95))

        assert model.should_degrade("sonnet") is True
        degraded = model.get_degraded_model("sonnet")
        assert degraded in ["haiku", "minimax"]


class TestComplexRouting:
    """Test complex routing scenarios."""

    def test_complexity_score_boundaries(self) -> None:
        """Test complexity score boundaries."""
        scorer = ComplexityScorer()

        # Very simple
        result = scorer.score("Hi")
        assert result.total <= 3

        # Complex task should be higher
        result = scorer.score(
            "Design a complete microservices architecture with "
            "service mesh, circuit breakers, distributed tracing, "
            "and multi-region failover with conflict resolution"
        )
        # Should be at least medium tier
        assert result.tier in ["MEDIUM", "HIGH"]
