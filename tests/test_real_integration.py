"""End-to-end integration tests - require real API keys.

Tests are split into:
- Mocked: Fast, deterministic tests with provider.chat() mocked
- Real API: Slow tests that call actual providers (marked @pytest.mark.real_api)
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()  # Load .env before checking API keys

# Check if API keys are available
SKIP_REASON = "API keys not set - skipping integration test"


def require_api_keys() -> bool:
    """Check if required API keys are set."""
    return bool(
        os.environ.get("MINIMAX_API_KEY")
        and os.environ.get("ANTHROPIC_API_KEY")
        and os.environ.get("LEMON_API_KEY")
    )


class MockLLMResponse:
    """Mock LLM response matching the provider response interface."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.model = "mock"
        self.usage: dict[str, int] = {"input_tokens": 10, "output_tokens": len(content) // 4}
        self.raw: dict[str, object] = {}
        self.finish_reason: str = "stop"
        self.tool_calls: list[dict[str, object]] | None = None


def mock_provider_chat(self: object, messages: list[dict[str, str]], model: str) -> MockLLMResponse:
    """Shared mock for all provider chat() methods."""
    last_msg = messages[-1]["content"] if messages else ""
    # Return deterministic responses based on prompt length/complexity
    if len(last_msg) < 20:
        return MockLLMResponse("Short response: 2+2=4.")
    if len(last_msg) < 100:
        return MockLLMResponse("Medium response about the topic.")
    return MockLLMResponse(
        "Detailed response: This is a comprehensive answer to the question. "
        "The system architecture involves multiple layers working together."
    )


# Only skip entire file if keys are missing
pytestmark = pytest.mark.skipif(not require_api_keys(), reason=SKIP_REASON)


# =============================================================================
# Fast mocked tests (always run when keys are present)
# =============================================================================


class TestL1Compression:
    """Test L1 compression and scoring (no network needed)."""

    def test_l1_compress_and_score(self) -> None:
        """Test L1 compression + complexity scoring."""
        from blend.intent.scorer import ComplexityScorer
        from blend.utils.compress import compress_prompt

        prompt = "设计一个高可用的分布式系统架构，包含负载均衡、数据库主从复制、缓存层和消息队列"

        scorer = ComplexityScorer()
        score = scorer.score(prompt)
        assert 1 <= score.total <= 10
        assert score.tier in ("LOW", "MEDIUM", "HIGH")

        result = compress_prompt(prompt)
        assert len(result.compressed) <= len(prompt)
        assert 0 <= result.compression_ratio <= 1.0


class TestL2StrategyMocked:
    """Test L2 strategy generation with mocked Opus."""

    def test_l2_strategy_generation_mocked(self) -> None:
        """Test L2 strategy generation mocked with JSON response."""
        from blend.core.strategy import StrategyGenerator

        _mock_response = MockLLMResponse(
            '{"plan": ["Step 1: Analyze", "Step 2: Implement"], '
            '"quality_redlines": ["No SQL injection"], '
            '"boundary_cases": ["Empty input"]]}'
        )

        with patch("blend.providers.baosiapi.BaosiProvider.chat", mock_provider_chat):
            generator = StrategyGenerator()
            result = generator.generate(
                prompt="Design a system",
                complexity=9,
            )

            assert result.output.plan is not None
            assert len(result.output.plan) > 0
            assert result.output.estimated_tokens <= 300


class TestL3ExecutionMocked:
    """Test L3 execution layer with mocked providers."""

    def test_l3_minimax_execution_mocked(self) -> None:
        """Test L3 execution with mocked Minimax (exhaust Haiku to force Minimax)."""
        from blend.core.executor import Executor

        with patch("blend.providers.minimax_new.MinimaxProvider.chat", mock_provider_chat):
            executor = Executor()
            # Exhaust haiku budget so Tier1 complexity=2 falls back to Minimax
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.side_effect = (
                lambda m: 0 if m == "haiku" else 10000
            )
            result = executor.execute(
                prompt="What is 2+2?",
                complexity=2,
            )

            assert result.raw_output is not None
            assert len(result.raw_output) > 0
            assert result.model_used == "minimax"
            assert result.tokens_used >= 0

    def test_l3_complexity_routing_mocked(self) -> None:
        """Test that complexity affects model selection (mocked)."""
        from blend.core.executor import Executor

        def mock_chat(self_: object, messages: list[dict[str, object]], model: str) -> MockLLMResponse:
            return MockLLMResponse("Mocked response content.")

        with patch.multiple(
            "blend.providers.minimax_new.MinimaxProvider",
            chat=mock_chat,
        ), patch.multiple(
            "blend.providers.baosiapi.BaosiProvider",
            chat=mock_chat,
        ):
            executor = Executor()
            # All budgets available: Tier1(2)→Haiku, Tier2(5)→Sonnet
            executor.resource_model = MagicMock()
            executor.resource_model.get_remaining.return_value = 10000

            result_low = executor.execute(prompt="Hi", complexity=2)
            assert result_low.model_used in ["haiku", "minimax"]

            result_med = executor.execute(
                prompt="Explain why the sky is blue in one paragraph",
                complexity=5,
            )
            assert result_med.model_used in ["haiku", "sonnet", "minimax"]


class TestL5Verification:
    """Test L5 verification layer (no network needed)."""

    def test_l5_quality_gate(self) -> None:
        """Test L5 quality gate verification."""
        from blend.core.verifier import QualityVerifier

        verifier = QualityVerifier()

        result = verifier.verify(
            output="This is a clean response about distributed systems.",
            quality_level="MEDIUM",
            layer_path="L1>L3>L5",
            output_tokens=100,
            l4_applied=False,
        )
        assert result.passed is True

        result = verifier.verify(
            output="This is a bad output with xxx content.",
            quality_level="LOW",
            layer_path="L1>L3>L5",
            output_tokens=100,
        )
        assert result.passed is False


class TestOrchestratorIntegrationMocked:
    """Test full orchestrator pipeline with mocked providers (fast, reliable)."""

    def test_orchestrator_simple_prompt_mocked(self) -> None:
        """Test orchestrator with simple prompt (mocked providers)."""
        from blend.core.orchestrator import BlendOrchestrator

        with patch("blend.providers.minimax_new.MinimaxProvider.chat", mock_provider_chat), \
             patch("blend.providers.baosiapi.BaosiProvider.chat", mock_provider_chat), \
             patch("blend.providers.lemonapi.LemonProvider.chat", mock_provider_chat):
            orchestrator = BlendOrchestrator()
            result = orchestrator.process("What is Python?")

            assert result.final_output is not None
            assert len(result.final_output) > 0
            assert "L1" in result.layer_path
            # Recipe architecture: MEDIUM uses DRAFT>REFINE or EXECUTE (gemini types use EXECUTE)
            assert "DRAFT" in result.layer_path or "REFINE" in result.layer_path or "EXECUTE" in result.layer_path or "L3" in result.layer_path
            assert "L5" in result.layer_path
            assert result.model_used in ("minimax", "haiku", "sonnet", "opus", "gemini")

    def test_orchestrator_complex_prompt_mocked(self) -> None:
        """Test orchestrator with complex prompt (mocked - eliminates flaky network calls)."""
        from blend.core.orchestrator import BlendOrchestrator

        with patch("blend.providers.minimax_new.MinimaxProvider.chat", mock_provider_chat), \
             patch("blend.providers.baosiapi.BaosiProvider.chat", mock_provider_chat), \
             patch("blend.providers.lemonapi.LemonProvider.chat", mock_provider_chat):
            orchestrator = BlendOrchestrator()
            result = orchestrator.process(
                "Design a microservices architecture with service mesh, "
                "observability, and disaster recovery for an e-commerce platform"
            )

            assert result.final_output is not None
            assert len(result.final_output) > 0
            assert result.complexity >= 5
            assert "L1" in result.layer_path
            # Recipe architecture: MEDIUM uses DRAFT>REFINE or EXECUTE (gemini types use EXECUTE)
            assert "DRAFT" in result.layer_path or "REFINE" in result.layer_path or "EXECUTE" in result.layer_path or "L3" in result.layer_path

    def test_orchestrator_high_complexity_includes_l2_mocked(self) -> None:
        """HIGH complexity orchestrator should include L2 in layer path."""
        from blend.core.orchestrator import BlendOrchestrator

        with patch("blend.providers.minimax_new.MinimaxProvider.chat", mock_provider_chat), \
             patch("blend.providers.baosiapi.BaosiProvider.chat", mock_provider_chat), \
             patch("blend.providers.lemonapi.LemonProvider.chat", mock_provider_chat), \
             patch("blend.intent.scorer.ComplexityScorer.score") as mock_score:
            mock_score.return_value = MagicMock(
                total=9,
                tier="HIGH",
                task_type="general",
                breakdown={},
                route_decision="HIGH",
            )

            orchestrator = BlendOrchestrator()
            result = orchestrator.process(
                "Design a microservices architecture with service mesh, "
                "observability, and disaster recovery for an e-commerce platform"
            )

            assert result.final_output is not None
            assert result.complexity == 9
            assert "L2" in result.layer_path


# =============================================================================
# Real API tests (slow, network-dependent — opt-in with -m real_api)
# =============================================================================


@pytest.mark.real_api
class TestL2StrategyRealAPI:
    """Test L2 strategy generation with real Opus API."""

    def test_l2_strategy_generation_real(self) -> None:
        """Test L2 strategy generation with real Opus call."""
        from blend.core.strategy import StrategyGenerator

        prompt = (
            "设计一个支持微服务架构的分布式系统，需要考虑服务发现、负载均衡、容错、监控等企业级需求"
        )
        generator = StrategyGenerator()

        from blend.intent.scorer import ComplexityScorer

        scorer = ComplexityScorer()
        score = scorer.score(prompt)

        if score.tier != "HIGH":
            pytest.skip("Prompt complexity not HIGH enough for L2 test")

        result = generator.generate(prompt=prompt, complexity=score.total)
        assert result.output.plan is not None
        assert len(result.output.plan) > 0
        assert result.output.estimated_tokens <= 300


@pytest.mark.real_api
class TestL3ExecutionRealAPI:
    """Test L3 execution layer with real API calls."""

    def test_l3_minimax_execution_real(self) -> None:
        """Test L3 execution with real Haiku API (Tier1=Haiku)."""
        from blend.core.executor import Executor

        executor = Executor()
        result = executor.execute(
            prompt="What is 2+2?",
            complexity=2,
        )

        assert result.raw_output is not None
        assert len(result.raw_output) > 0
        assert result.model_used in ["haiku", "minimax"]
        assert result.tokens_used > 0

    def test_l3_complexity_routing_real(self) -> None:
        """Test that complexity affects model selection (real API)."""
        from blend.core.executor import Executor

        executor = Executor()

        result_low = executor.execute(prompt="Hi", complexity=2)
        # haiku may be unhealthy → gemini is valid fallback; minimax always available
        assert result_low.model_used in ["haiku", "gemini", "minimax"]

        result_med = executor.execute(
            prompt="Explain why the sky is blue in one paragraph",
            complexity=5,
        )
        assert result_med.model_used in ["haiku", "sonnet", "minimax", "gemini"]


@pytest.mark.real_api
class TestOrchestratorIntegrationRealAPI:
    """Test full orchestrator pipeline with real API calls."""

    def test_orchestrator_simple_prompt_real(self) -> None:
        """Test orchestrator with simple prompt (real API).

        Note: Marked real_api - may timeout or fail if API is unavailable.
        """
        import httpx
        from blend.core.orchestrator import BlendOrchestrator

        try:
            orchestrator = BlendOrchestrator()
            result = orchestrator.process("What is Python?")
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            # Network/API instability - not a code bug
            pytest.skip(f"Real API unavailable: {type(e).__name__}")

        assert result.final_output is not None
        assert len(result.final_output) > 0
        assert "L1" in result.layer_path
        # Recipe architecture: may use EXECUTE stage instead of L3 for LOW complexity routing
        assert "EXECUTE" in result.layer_path or "L3" in result.layer_path
        assert "L5" in result.layer_path
        assert result.model_used in ("minimax", "haiku", "sonnet", "opus", "gemini")

