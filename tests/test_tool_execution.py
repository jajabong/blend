"""Tests for agentic tool execution loop (v1.4)."""

from unittest.mock import MagicMock, patch

from blend.core.orchestrator import BlendOrchestrator, OrchestratorResult


class TestToolExecutorUnit:
    """Unit tests for tool_executor module."""

    def test_execute_single_tool_calculator_success(self) -> None:
        """calculator tool returns correct result."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'},
        }
        result = execute_single_tool(tool_call)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_abc"
        assert result["content"] == "4"

    def test_execute_single_tool_calculator_complex(self) -> None:
        """calculator handles complex expressions."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_xyz",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expr": "(10 + 5) * 3"}'},
        }
        result = execute_single_tool(tool_call)
        assert result["content"] == "45"

    def test_execute_single_tool_unknown(self) -> None:
        """Unknown tool returns error result."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_unknown",
            "type": "function",
            "function": {"name": "nonexistent_tool", "arguments": "{}"},
        }
        result = execute_single_tool(tool_call)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_unknown"
        assert "Unknown tool" in result["content"]

    def test_execute_single_tool_malformed_json(self) -> None:
        """Malformed JSON in arguments returns error result."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_bad",
            "type": "function",
            "function": {"name": "calculator", "arguments": "not valid json"},
        }
        result = execute_single_tool(tool_call)
        assert "Invalid arguments JSON" in result["content"]

    def test_execute_single_tool_defined_but_not_registered(self) -> None:
        """Tool defined in tools list but not locally registered returns informative error."""
        from blend.core.tool_executor import execute_single_tool

        tool_call = {
            "id": "call_remote",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        result = execute_single_tool(tool_call, tools=tools)
        assert "not registered for local execution" in result["content"]

    def test_execute_tool_calls_parallel(self) -> None:
        """execute_tool_calls runs multiple tools and returns results in order."""
        from blend.core.tool_executor import execute_tool_calls

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expr": "3*3"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expr": "10-4"}'},
            },
        ]
        results = execute_tool_calls(tool_calls)
        assert len(results) == 2
        assert results[0]["content"] == "9"
        assert results[1]["content"] == "6"


class TestOrchestratorToolLoop:
    """Test process_messages tool execution loop."""

    def test_process_messages_single_tool_call(self) -> None:
        """Orchestrator executes a tool call and returns final result."""
        orchestrator = BlendOrchestrator()

        with patch.object(orchestrator.scorer, "score") as mock_score, \
             patch.object(orchestrator.executor, "execute_messages") as mock_exec, \
             patch.object(orchestrator.verifier, "verify") as mock_verify, \
             patch.object(orchestrator.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(
                total=3, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )

            # First call: model wants to call calculator
            first_output = MagicMock(
                content="Let me calculate that.",
                model_used="minimax",
                tokens_used=10,
                quality_gate_passed=True,
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'},
                }],
            )
            # Second call: model returns stop
            second_output = MagicMock(
                content="2+2 equals 4.",
                model_used="minimax",
                tokens_used=15,
                quality_gate_passed=True,
                finish_reason="stop",
                tool_calls=None,
            )
            mock_exec.side_effect = [first_output, second_output]
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            result = orchestrator.process_messages(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Math",
                        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
                    },
                }],
            )

            # Should have called executor twice
            assert mock_exec.call_count == 2

            # Final result should be from second call
            assert result.final_output == "2+2 equals 4."
            assert result.finish_reason == "stop"
            assert result.tool_call_count == 1

    def test_process_messages_no_tools_terminates_immediately(self) -> None:
        """Without tools param, orchestrator returns without looping."""
        orchestrator = BlendOrchestrator()

        with patch.object(orchestrator.scorer, "score") as mock_score, \
             patch.object(orchestrator.executor, "execute_messages") as mock_exec, \
             patch.object(orchestrator.verifier, "verify") as mock_verify, \
             patch.object(orchestrator.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(
                total=2, tier="LOW", task_type="general",
                breakdown={}, route_decision="LOW",
            )
            mock_exec.return_value = MagicMock(
                content="Hello!",
                model_used="minimax",
                tokens_used=5,
                quality_gate_passed=True,
                finish_reason="stop",
                tool_calls=None,
            )
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            result = orchestrator.process_messages(
                messages=[{"role": "user", "content": "Hi"}],
            )

            # Should only call executor once (no loop)
            assert mock_exec.call_count == 1
            assert result.finish_reason == "stop"
            assert result.tool_call_count == 0

    def test_process_messages_max_iterations_guard(self) -> None:
        """Loop stops at MAX_TOOL_ITERATIONS even if model keeps requesting tools."""
        orchestrator = BlendOrchestrator()
        max_iter = 10  # must match orchestrator

        with patch.object(orchestrator.scorer, "score") as mock_score, \
             patch.object(orchestrator.executor, "execute_messages") as mock_exec, \
             patch.object(orchestrator.verifier, "verify") as mock_verify, \
             patch.object(orchestrator.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(
                total=3, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            # Always return tool_calls
            mock_exec.return_value = MagicMock(
                content="Calling tool...",
                model_used="minimax",
                tokens_used=10,
                quality_gate_passed=True,
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expr": "1+1"}'},
                }],
            )
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            orchestrator.process_messages(
                messages=[{"role": "user", "content": "Calculate"}],
                tools=[{
                    "type": "function",
                    "function": {"name": "calculator", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}},
                }],
            )

            # Note: orchestrator makes max_iter calls in loop + 1 final synthesis call
            # when tools were executed and finish_reason="tool_calls"
            assert mock_exec.call_count == max_iter + 1

    def test_process_messages_messages_accumulate(self) -> None:
        """After tool execution, messages list contains: user + assistant(tool_calls) + tool results."""
        orchestrator = BlendOrchestrator()

        with patch.object(orchestrator.scorer, "score") as mock_score, \
             patch.object(orchestrator.executor, "execute_messages") as mock_exec, \
             patch.object(orchestrator.verifier, "verify") as mock_verify, \
             patch.object(orchestrator.enforcer, "enforce") as mock_enforce:

            mock_score.return_value = MagicMock(
                total=3, tier="MEDIUM", task_type="general",
                breakdown={}, route_decision="MEDIUM",
            )
            first_output = MagicMock(
                content="Computing...",
                model_used="minimax",
                tokens_used=8,
                quality_gate_passed=True,
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expr": "5*5"}'},
                }],
            )
            second_output = MagicMock(
                content="5 times 5 is 25.",
                model_used="minimax",
                tokens_used=12,
                quality_gate_passed=True,
                finish_reason="stop",
                tool_calls=None,
            )
            mock_exec.side_effect = [first_output, second_output]
            mock_verify.return_value = MagicMock(passed=True)
            mock_enforce.return_value = MagicMock(allowed=True, violations=[])

            orchestrator.process_messages(
                messages=[{"role": "user", "content": "What is 5*5?"}],
                tools=[{
                    "type": "function",
                    "function": {"name": "calculator", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}},
                }],
            )

            # Check that executor was called with updated messages
            assert mock_exec.call_count == 2

            # The mutable `current_messages` list is mutated in-place between calls.
            # Both call_args_list entries reference the same list object (post-mutation).
            # Verify the final state has 3 messages.
            final_call_msgs = mock_exec.call_args_list[1].kwargs["messages"]
            assert len(final_call_msgs) == 3
            assert final_call_msgs[0]["role"] == "user"
            assert final_call_msgs[1]["role"] == "assistant"
            assert "tool_calls" in final_call_msgs[1]
            assert final_call_msgs[2]["role"] == "tool"
            assert final_call_msgs[2]["tool_call_id"] == "call_1"


class TestOrchestratorResultNewFields:
    """Test OrchestratorResult has new tool tracking fields."""

    def test_tool_call_count_field_exists(self) -> None:
        """OrchestratorResult accepts tool_call_count and tool_loop_iterations."""
        result = OrchestratorResult(
            final_output="test",
            layer_path="L1>L3>L5",
            complexity=3,
            model_used="minimax",
            tokens_used=10,
            quality_gate_passed=True,
            l1_compressed=False,
            l4_applied=False,
            tool_call_count=5,
            tool_loop_iterations=3,
        )
        assert result.tool_call_count == 5
        assert result.tool_loop_iterations == 3

    def test_tool_call_count_defaults_to_zero(self) -> None:
        """tool_call_count and tool_loop_iterations default to 0."""
        result = OrchestratorResult(
            final_output="hello",
            layer_path="L1>L3>L5",
            complexity=1,
            model_used="minimax",
            tokens_used=5,
            quality_gate_passed=True,
            l1_compressed=False,
            l4_applied=False,
        )
        assert result.tool_call_count == 0
        assert result.tool_loop_iterations == 0
