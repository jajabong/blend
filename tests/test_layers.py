"""Tests for blend core layers."""

from blend.core.layers import (
    L1Output,
    L2Output,
    L3Output,
    L4Output,
    L5Output,
    Layer,
)


class TestLayer:
    """Test Layer enum."""

    def test_layer_values(self) -> None:
        """Test all layer values exist."""
        assert Layer.L1_ENTRY.value == "L1"
        assert Layer.L2_STRATEGY.value == "L2"
        assert Layer.L3_EXECUTE.value == "L3"
        assert Layer.L4_COMPRESS.value == "L4"
        assert Layer.L5_VERIFY.value == "L5"


class TestL1Output:
    """Test L1 output structure."""

    def test_l1_output_creation(self) -> None:
        """Test creating L1Output."""
        output = L1Output(
            compressed_prompt="test prompt",
            complexity_score=5,
            complexity_breakdown={"steps": 1, "domain": 2, "output": 1, "creativity": 1, "risk": 0},
            route_decision="L3_HAIKU",
            l1_compressed=True,
            compression_ratio=0.65,
        )
        assert output.compressed_prompt == "test prompt"
        assert output.complexity_score == 5
        assert output.route_decision == "L3_HAIKU"
        assert output.l1_compressed is True


class TestL2Output:
    """Test L2 output structure."""

    def test_l2_output_creation(self) -> None:
        """Test creating L2Output."""
        output = L2Output(
            plan=["Step 1", "Step 2"],
            quality_redlines=["No hardcoded secrets"],
            boundary_cases=["Empty input"],
            model_hint="Sonnet",
            estimated_tokens=150,
        )
        assert len(output.plan) == 2
        assert output.model_hint == "Sonnet"
        assert output.estimated_tokens == 150


class TestL3Output:
    """Test L3 output structure."""

    def test_l3_output_creation(self) -> None:
        """Test creating L3Output."""
        output = L3Output(
            raw_output="Response from model",
            model_used="haiku",
            tokens_used=500,
            tokens_budget_remaining=4500,
            quality_gate_passed=True,
        )
        assert output.model_used == "haiku"
        assert output.tokens_used == 500
        assert output.quality_gate_passed is True


class TestL4Output:
    """Test L4 output structure."""

    def test_l4_output_creation(self) -> None:
        """Test creating L4Output."""
        output = L4Output(
            compressed_output="Compressed response",
            original_tokens=1000,
            compressed_tokens=200,
            compression_ratio=0.8,
        )
        assert output.original_tokens == 1000
        assert output.compressed_tokens == 200
        assert output.compression_ratio == 0.8


class TestL5Output:
    """Test L5 output structure."""

    def test_l5_output_creation(self) -> None:
        """Test creating L5Output."""
        output = L5Output(
            final_output="Final verified output",
            quality_gate_passed=True,
            gates_checked={
                "no_taboo_violation": True,
                "format_compliant": True,
                "no_p0_vuln": True,
            },
            quality_level="HIGH",
            rejection_reason=None,
        )
        assert output.quality_gate_passed is True
        assert output.gates_checked["no_taboo_violation"] is True
        assert output.rejection_reason is None

    def test_l5_output_rejection(self) -> None:
        """Test L5Output with rejection."""
        output = L5Output(
            final_output="",
            quality_gate_passed=False,
            gates_checked={"no_taboo_violation": False},
            quality_level="LOW",
            rejection_reason="Taboo content detected",
        )
        assert output.quality_gate_passed is False
        assert output.rejection_reason == "Taboo content detected"
