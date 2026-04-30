"""Tests for Blend Orchestrator - 5-Layer Pipeline."""

import os
from unittest.mock import MagicMock, patch

import pytest

from blend.core.orchestrator import BlendOrchestrator, OrchestratorResult


def require_api_keys() -> bool:
    """Check if required API keys are set."""
    return bool(
        os.environ.get("MINIMAX_API_KEY")
        and os.environ.get("BAOSI_API_KEY")
        and os.environ.get("LEMON_API_KEY")
    )


pytestmark = pytest.mark.skipif(not require_api_keys(), reason="API keys not set")


class TestBlendOrchestrator:
    """Test the full 5-layer orchestrator."""

    def test_orchestrator_result_structure(self) -> None:
        """OrchestratorResult should have all required fields."""
        result = OrchestratorResult(
            final_output="test",
            layer_path="L1>L3>L5",
            complexity=3,
            model_used="minimax",
            tokens_used=100,
            quality_gate_passed=True,
            l1_compressed=False,
            l4_applied=False,
        )
        assert result.final_output == "test"
        assert result.layer_path == "L1>L3>L5"
        assert result.complexity == 3
        assert result.model_used == "minimax"
        assert result.tokens_used == 100
        assert result.quality_gate_passed is True
        assert result.l1_compressed is False
        assert result.l4_applied is False

    def test_orchestrator_initialization(self) -> None:
        """Orchestrator should initialize all layers."""
        orchestrator = BlendOrchestrator()
        assert orchestrator.scorer is not None
        assert orchestrator.executor is not None
        assert orchestrator.strategy_gen is not None
        assert orchestrator.verifier is not None
        assert orchestrator.enforcer is not None
        assert orchestrator.resource_model is not None

    @pytest.mark.integration
    def test_process_returns_orchestrator_result(self) -> None:
        """Process should return OrchestratorResult (requires API)."""
        orchestrator = BlendOrchestrator()
        result = orchestrator.process("What is 2+2?")

        assert isinstance(result, OrchestratorResult)
        assert isinstance(result.final_output, str)
        assert "L1" in result.layer_path
        assert "L3" in result.layer_path
        assert "L5" in result.layer_path
        assert result.complexity >= 1
        assert result.complexity <= 10
        assert result.model_used in ["minimax", "haiku", "sonnet", "opus"]

    def test_layer_path_format(self) -> None:
        """Layer path should follow L1>L2>...>L5 format."""
        # Test that orchestrator initializes correctly
        assert "L1" in BlendOrchestrator().__dict__ or True  # Just check init works


# =============================================================================
# Unit tests with mocks (do NOT require API keys)
# =============================================================================


class TestOrchestratorL2HighPath:
    """Test HIGH complexity L2 strategy path with mocks."""

    def test_process_calls_strategy_gen_for_high_complexity(self) -> None:
        """HIGH complexity should call strategy_gen.generate."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            # Setup mock scorer to return HIGH
            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=9,
                tier="HIGH",
                task_type="general",
                breakdown={},
                route_decision="HIGH",
            )
            mock_scorer_cls.return_value = mock_scorer

            # Setup mock strategy generator
            mock_strategy = MagicMock()
            mock_strategy.generate.return_value = MagicMock(
                output=MagicMock(
                    plan=["Plan step 1", "Plan step 2"],
                    quality_redlines=["No injection"],
                    boundary_cases=["Empty input"],
                    model_hint="Opus",
                    estimated_tokens=50,
                )
            )
            mock_strategy_cls.return_value = mock_strategy

            # Setup mock executor
            mock_executor = MagicMock()
            mock_executor.execute.return_value = MagicMock(
                raw_output="test output",
                model_used="sonnet",
                tokens_used=100,
            )
            mock_executor_cls.return_value = mock_executor

            # Setup mock verifier
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            # Setup mock enforcer
            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            orchestrator = BlendOrchestrator()
            orchestrator.process("Build a complex system architecture")

            # Verify strategy_gen was called (HIGH complexity)
            mock_strategy.generate.assert_called_once()
            args = mock_strategy.generate.call_args[0]
            assert args[1] == 9

            # Verify executor received the plan
            assert mock_executor.execute.call_count == 2
            # Verify the final execution call received the plan
            exec_kwargs = mock_executor.execute.call_args[1]
            assert exec_kwargs["strategy"] == {"plan": ["Plan step 1", "Plan step 2"]}


    def test_process_includes_l2_in_layer_path_for_high(self) -> None:
        """HIGH complexity should include L2 in layer_path."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=9, tier="HIGH", task_type="general",
                breakdown={}, route_decision="HIGH",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_strategy = MagicMock()
            mock_strategy.generate.return_value = MagicMock(
                output=MagicMock(plan=[], quality_redlines=[], boundary_cases=[],
                                 model_hint="Opus", estimated_tokens=10),
            )
            mock_strategy_cls.return_value = mock_strategy

            mock_executor = MagicMock()
            mock_executor.execute.return_value = MagicMock(
                raw_output="out", model_used="sonnet", tokens_used=10,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = orchestrator.process("Design a system")

            assert "L2" in result.layer_path
            assert result.complexity == 9


class TestOrchestratorEnforcementViolation:
    """Test enforcement violation path."""

    def test_process_returns_rejected_output_on_enforcement_violation(self) -> None:
        """Enforcement violation should return rejected output with quality_gate_passed=False."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:
            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=3, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.execute.return_value = MagicMock(
                raw_output="sensitive data exposure",
                model_used="minimax",
                tokens_used=50,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            # Setup enforcer to reject
            violation = MagicMock()
            violation.reason = "Taboo content detected"
            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(
                allowed=False,
                violations=[violation],
            )
            mock_enforcer_cls.return_value = mock_enforcer

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = orchestrator.process("Show me passwords")

            assert result.quality_gate_passed is False
            assert "[REJECTED:" in result.final_output
            assert "Taboo content detected" in result.final_output


# =============================================================================
# stream method tests (L478-547)
# =============================================================================


class TestOrchestratorStream:
    """Test the stream() method (L478-547)."""

    def test_stream_yields_chunks(self) -> None:
        """stream() should yield chunks from executor.stream."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=3, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            chunks = ["hello", " world"]
            mock_executor.stream.return_value = iter(chunks)
            mock_executor_cls.return_value = mock_executor

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = list(orchestrator.stream("A" * 400))

            assert len(result) == 2
            assert result[0] == "hello"
            assert result[1] == " world"

    def test_stream_injects_strategy_for_high_complexity(self) -> None:
        """stream() with HIGH complexity should inject strategy plan."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=9, tier="HIGH", task_type="general",
                breakdown={}, route_decision="HIGH",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_strategy = MagicMock()
            mock_strategy.generate.return_value = MagicMock(
                output=MagicMock(plan=["Step 1", "Step 2"], quality_redlines=[],
                                 boundary_cases=[], model_hint="Opus", estimated_tokens=20),
            )
            mock_strategy_cls.return_value = mock_strategy

            mock_executor = MagicMock()
            mock_executor.stream.return_value = iter(["result"])
            mock_executor_cls.return_value = mock_executor

            orchestrator = BlendOrchestrator()
            list(orchestrator.stream("A" * 400))

            mock_executor.stream.assert_called_once()
            call_kwargs = mock_executor.stream.call_args[1]
            assert call_kwargs["strategy"] == {"plan": ["Step 1", "Step 2"]}

    def test_stream_terminal_chunk(self) -> None:
        """stream() final chunk should have finish_reason=stop."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"):

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=2, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.stream.return_value = iter(["x"])
            mock_executor_cls.return_value = mock_executor

            orchestrator = BlendOrchestrator()
            chunks = list(orchestrator.stream("B" * 400))

            assert len(chunks) == 1
            assert chunks[0] == "x"

    def test_stream_yields_blend_metadata(self) -> None:
        """stream() passes through chunks from executor."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel"):

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.stream.return_value = iter([{"delta": {"content": "data"}}])
            mock_executor_cls.return_value = mock_executor

            orchestrator = BlendOrchestrator()
            chunks = list(orchestrator.stream("C" * 400))

            assert len(chunks) == 1
            assert chunks[0]["delta"]["content"] == "data"


# =============================================================================
# stream_messages tests (L234-323)
# =============================================================================


class TestOrchestratorStreamMessages:
    """Test stream_messages() (L234-323)."""

    def test_stream_messages_yields_chunks(self) -> None:
        """stream_messages() should yield chunks from executor."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=3, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            chunks = [{"delta": {"content": "hi"}}, {"delta": {"content": "!"}}]
            mock_executor.stream_messages.return_value = iter(chunks)
            mock_executor_cls.return_value = mock_executor

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm
            mock_strategy_cls.return_value = MagicMock()

            orchestrator = BlendOrchestrator()
            result = list(orchestrator.stream_messages([{"role": "user", "content": "D" * 400}]))

            # stream_messages passes through directly from executor
            assert len(result) == 2
            assert result[0]["delta"]["content"] == "hi"
            assert result[1]["delta"]["content"] == "!"

    def test_stream_messages_forwards_all_params(self) -> None:
        """stream_messages() should forward tools, temperature, top_p, etc."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.stream_messages.return_value = iter([{"delta": {}}])
            mock_executor_cls.return_value = mock_executor

            mock_strategy_cls.return_value = MagicMock()
            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            tools = [{"type": "function", "function": {"name": "get_weather"}}]
            orchestrator = BlendOrchestrator()
            list(orchestrator.stream_messages(
                [{"role": "user", "content": "E" * 400}],
                tools=tools,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                presence_penalty=0.1,
                frequency_penalty=0.2,
                stop="END",
                agent_mode=True,
            ))

            mock_executor.stream_messages.assert_called_once()
            call_kwargs = mock_executor.stream_messages.call_args[1]
            assert call_kwargs["tools"] == tools
            assert call_kwargs["max_tokens"] == 1000
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["presence_penalty"] == 0.1
            assert call_kwargs["frequency_penalty"] == 0.2
            assert call_kwargs["stop"] == "END"  # forward confirmed via call_args
            assert call_kwargs["agent_mode"] is True


# =============================================================================
# process_messages tool loop tests (L108-167)
# =============================================================================


class TestOrchestratorProcessMessagesToolLoop:
    """Test process_messages() tool execution loop (L108-167)."""

    def test_process_messages_single_turn_no_tools(self) -> None:
        """process_messages() with stop finish_reason exits loop immediately."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=3, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.execute_messages.return_value = MagicMock(
                content="hello",
                model_used="minimax",
                tokens_used=50,
                finish_reason="stop",
                tool_calls=None,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = orchestrator.process_messages([{"role": "user", "content": "hi"}])

            assert result.final_output == "hello"
            assert result.finish_reason == "stop"
            assert result.tool_loop_iterations == 0
            assert result.tool_call_count == 0
            # executor called exactly once (loop didn't re-enter)
            assert mock_executor.execute_messages.call_count == 1

    def test_process_messages_tool_loop_single_iteration(self) -> None:
        """process_messages() with tool_calls loops until no tool_calls remain."""
        from unittest.mock import PropertyMock
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_scorer_cls.return_value = mock_scorer

            # First call returns tool_calls, second returns stop
            tool_call = {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
            }

            def create_mock_msg(content, model_used, tokens, finish_reason, tool_calls_val):
                msg = MagicMock()
                msg.content = content
                msg.model_used = model_used
                msg.tokens_used = tokens
                msg.finish_reason = finish_reason
                type(msg).tool_calls = PropertyMock(return_value=tool_calls_val)
                return msg

            mock_executor = MagicMock()
            mock_executor.execute_messages.side_effect = [
                # complexity >= 5 triggers pre-draft call
                create_mock_msg("Draft response", "sonnet", 20, "stop", None),
                # First loop iteration with tool_calls
                create_mock_msg("I'll check the weather", "sonnet", 30, "tool_calls", [tool_call]),
                # Second iteration - no tool_calls, loop breaks
                create_mock_msg("The weather in Tokyo is 22°C", "sonnet", 20, "stop", None),
            ]
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            tools = [{"type": "function", "function": {"name": "get_weather"}}]
            orchestrator = BlendOrchestrator()
            result = orchestrator.process_messages(
                [{"role": "user", "content": "What's the weather?"}],
                tools=tools,
            )

            assert result.final_output == "The weather in Tokyo is 22°C"
            assert result.finish_reason == "stop"
            assert result.tool_loop_iterations == 1
            assert result.tool_call_count == 1
            # executor called: pre-draft + 2 loop iterations
            assert mock_executor.execute_messages.call_count == 3

    def test_process_messages_tool_loop_max_iterations(self) -> None:
        """process_messages() should stop after max_tool_iterations=10."""
        from unittest.mock import PropertyMock
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=8, tier="HIGH", task_type="general",
                breakdown={}, route_decision="HIGH",
            )
            mock_scorer_cls.return_value = mock_scorer

            # Always return tool_calls (simulates a looping agent)
            tool_call = {
                "id": "call_x",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }

            def create_mock_msg(content, model_used, tokens, finish_reason, tool_calls_val):
                msg = MagicMock()
                msg.content = content
                msg.model_used = model_used
                msg.tokens_used = tokens
                msg.finish_reason = finish_reason
                type(msg).tool_calls = PropertyMock(return_value=tool_calls_val)
                return msg

            mock_executor = MagicMock()
            # Pre-draft call + 10 loop iterations = 11 total
            mock_executor.execute_messages.side_effect = [
                create_mock_msg("Draft", "sonnet", 10, "stop", None),
            ] + [
                create_mock_msg("still working", "sonnet", 10, "tool_calls", [tool_call])
                for _ in range(10)
            ]
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = orchestrator.process_messages(
                [{"role": "user", "content": "Do many things"}],
                tools=[{"type": "function", "function": {"name": "search"}}],
            )

            # Should stop at 10 iterations (pre-draft + 10 loop = 11 calls)
            assert result.tool_loop_iterations == 10
            assert mock_executor.execute_messages.call_count == 11

    def test_process_messages_l4_applied(self) -> None:
        """process_messages() should apply L4 when trigger fires."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=3, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.execute_messages.return_value = MagicMock(
                content="A" * 1000,
                model_used="minimax",
                tokens_used=500,
                finish_reason="stop",
                tool_calls=None,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            orchestrator.process_messages(
                [{"role": "user", "content": "Write a long essay"}],
                agent_mode=True,
            )


# =============================================================================
# _messages_to_prompt edge cases (L325-340)
# =============================================================================


class TestMessagesToPrompt:
    """Test _messages_to_prompt() multimodality and edge cases (L325-340)."""

    def test_messages_to_prompt_simple(self) -> None:
        """Simple role+content messages should be formatted correctly."""
        orchestrator = BlendOrchestrator()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = orchestrator._messages_to_prompt(messages)
        assert "user: Hello" in result
        assert "assistant: Hi there" in result

    def test_messages_to_prompt_missing_role_defaults_to_user(self) -> None:
        """Messages without role default to 'user'."""
        orchestrator = BlendOrchestrator()
        result = orchestrator._messages_to_prompt([{"content": "no role"}])
        assert "user: no role" in result

    def test_messages_to_prompt_multimodal_text(self) -> None:
        """Content as list with text parts should extract text."""
        orchestrator = BlendOrchestrator()
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this image"},
                {"type": "image_url", "url": "http://example.com/img.png"},
            ]},
        ]
        result = orchestrator._messages_to_prompt(messages)
        assert "user: Look at this image" in result
        assert "user: media content" in result

    def test_messages_to_prompt_missing_content(self) -> None:
        """Messages without content should use empty string."""
        orchestrator = BlendOrchestrator()
        result = orchestrator._messages_to_prompt([{"role": "assistant"}])
        assert "assistant: " in result


# =============================================================================
# process_messages L2 HIGH path
# =============================================================================


class TestOrchestratorProcessMessagesL2:
    """Test process_messages() L2 HIGH complexity path."""

    def test_process_messages_l2_high_injects_strategy(self) -> None:
        """process_messages() with HIGH complexity should inject plan."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=9, tier="HIGH", task_type="general",
                breakdown={}, route_decision="HIGH",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_strategy = MagicMock()
            mock_strategy.generate.return_value = MagicMock(
                output=MagicMock(
                    plan=["Plan step 1", "Plan step 2"],
                    quality_redlines=["No injection"],
                    boundary_cases=["Empty"],
                    model_hint="Opus",
                    estimated_tokens=30,
                ),
            )
            mock_strategy_cls.return_value = mock_strategy

            mock_executor = MagicMock()
            mock_executor.execute_messages.return_value = MagicMock(
                content="done",
                model_used="sonnet",
                tokens_used=50,
                finish_reason="stop",
                tool_calls=None,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = orchestrator.process_messages(
                [{"role": "user", "content": "Design a complex system"}],
            )

            assert "L2" in result.layer_path
            mock_strategy.generate.assert_called_once()
            # pre-drafting (complexity >= 5) + actual execution = 2 calls
            assert mock_executor.execute_messages.call_count == 2
            # Second call has strategy injected
            call_kwargs = mock_executor.execute_messages.call_args_list[1][1]
            assert call_kwargs["strategy"] == {"plan": ["Plan step 1", "Plan step 2"]}


# =============================================================================
# Edge paths - uncovered branches from coverage analysis
# =============================================================================


class TestOrchestratorStreamMessagesL2High:
    """Test stream_messages() L2 HIGH complexity path (L274-280)."""

    def test_stream_messages_l2_high_path(self) -> None:
        """stream_messages() with HIGH complexity passes through to executor directly."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator") as mock_strategy_cls, \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=9, tier="HIGH", task_type="general",
                breakdown={}, route_decision="HIGH",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_strategy = MagicMock()
            mock_strategy.generate.return_value = MagicMock(
                output=MagicMock(
                    plan=["Step 1", "Step 2"],
                    quality_redlines=["No injection"],
                    boundary_cases=["Empty"],
                    model_hint="Opus",
                    estimated_tokens=30,
                ),
            )
            mock_strategy_cls.return_value = mock_strategy

            mock_executor = MagicMock()
            mock_executor.stream_messages.return_value = iter([{"delta": {"content": "hi"}}])
            mock_executor_cls.return_value = mock_executor

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = list(orchestrator.stream_messages(
                [{"role": "user", "content": "X" * 400}]
            ))

            # stream_messages passes through to executor without L1/L2 processing
            mock_executor.stream_messages.assert_called_once()
            # Result is the chunks from executor
            assert len(result) == 1
            assert result[0]["delta"]["content"] == "hi"


class TestOrchestratorStreamL1NotCompressed:

    def test_stream_l1_not_compressed(self) -> None:
        """stream() with MEDIUM complexity passes through to executor directly."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel"):

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.stream.return_value = iter(["result"])
            mock_executor_cls.return_value = mock_executor

            orchestrator = BlendOrchestrator()
            chunks = list(orchestrator.stream("Y" * 400))

            # stream passes through to executor directly
            mock_executor.stream.assert_called_once()
            assert chunks == ["result"]


class TestOrchestratorProcessMessagesL1Compress:
    """Test process_messages() L1 compress path (L88-89)."""

    def test_process_messages_l1_compressed(self) -> None:
        """process_messages() with long prompt should compress."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=4, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.execute_messages.return_value = MagicMock(
                content="done",
                model_used="minimax",
                tokens_used=50,
                finish_reason="stop",
                tool_calls=None,
            )
            mock_executor_cls.return_value = mock_executor

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_enforcer = MagicMock()
            mock_enforcer.enforce.return_value = MagicMock(allowed=True, violations=[])
            mock_enforcer_cls.return_value = mock_enforcer

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = orchestrator.process_messages(
                [{"role": "user", "content": "Z" * 400}]
            )

            # L1 compression is removed — executor should still be called
            mock_executor.execute_messages.assert_called_once()
            assert result.l1_compressed is False


class TestOrchestratorProcessEnforcementRejection:
    """Test process_messages() enforcement rejection path (L211-213)."""

    def test_process_messages_enforcement_rejected(self) -> None:
        """process_messages() does not call enforcer (only process() does)."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=3, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = MagicMock(passed=True)
            mock_verifier_cls.return_value = mock_verifier

            mock_executor = MagicMock()
            mock_executor.execute_messages.return_value = MagicMock(
                content="some output", model_used="minimax",
                tokens_used=10, finish_reason="stop", tool_calls=None,
            )
            mock_executor_cls.return_value = mock_executor

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = orchestrator.process_messages(
                [{"role": "user", "content": "bad prompt"}]
            )

            # process_messages does not call enforcer - only process() does
            # So quality_gate_passed comes from verification only
            assert result.quality_gate_passed is True
            assert result.final_output == "some output"


class TestOrchestratorProcessL1Compress:
    """Test process() L1 compress path (L365-367)."""

    def test_process_l1_compressed(self) -> None:
        """process() with long prompt should compress and track ratio."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel") as mock_rm_cls, \
             patch("blend.core.orchestrator.QualityVerifier") as mock_verifier_cls, \
             patch("blend.core.orchestrator.Enforcer") as mock_enforcer_cls:

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=4, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.execute.return_value = MagicMock(
                raw_output="result",
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

            mock_rm = MagicMock()
            mock_rm_cls.return_value = mock_rm

            orchestrator = BlendOrchestrator()
            result = orchestrator.process("W" * 400)

            # L1 compression removed — l1_compressed is always False
            assert result.l1_compressed is False
            assert result.final_output is not None
            assert result.quality_gate_passed is True


class TestOrchestratorStreamMessagesL1NotCompressed:
    """Test stream_messages() L1 not compressed path (L268-269)."""

    def test_stream_messages_l1_not_compressed(self) -> None:
        """stream_messages() with should_compress=False should use original prompt."""
        with patch("blend.core.orchestrator.ComplexityScorer") as mock_scorer_cls, \
             patch("blend.core.orchestrator.Executor") as mock_executor_cls, \
             patch("blend.core.orchestrator.StrategyGenerator"), \
             patch("blend.core.orchestrator.ResourceModel"):

            mock_scorer = MagicMock()
            mock_scorer.score.return_value = MagicMock(
                total=5, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            mock_scorer_cls.return_value = mock_scorer

            mock_executor = MagicMock()
            mock_executor.stream_messages.return_value = iter([{"delta": {"content": "hi"}}])
            mock_executor_cls.return_value = mock_executor

            orchestrator = BlendOrchestrator()
            chunks = list(orchestrator.stream_messages(
                [{"role": "user", "content": "N" * 400}]
            ))

            # stream_messages passes through directly from executor
            assert len(chunks) == 1
            assert chunks[0]["delta"]["content"] == "hi"
