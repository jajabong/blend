"""Comprehensive tests for Orchestrator to improve coverage."""

from unittest.mock import MagicMock

from blend.core.orchestrator import BlendOrchestrator


class TestOrchestratorMessagesToPrompt:
    """Test _messages_to_prompt method."""

    def test_empty_messages(self) -> None:
        """Empty messages return empty string."""
        orchestrator = BlendOrchestrator()
        result = orchestrator._messages_to_prompt([])
        assert result == ""

    def test_single_message(self) -> None:
        """Single message is converted."""
        orchestrator = BlendOrchestrator()
        messages = [{"role": "user", "content": "Hello"}]
        result = orchestrator._messages_to_prompt(messages)
        assert "Hello" in result

    def test_multiple_messages(self) -> None:
        """Multiple messages are concatenated."""
        orchestrator = BlendOrchestrator()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = orchestrator._messages_to_prompt(messages)
        assert "Hello" in result
        assert "Hi there" in result
        assert "How are you?" in result


class TestOrchestratorStream:
    """Test stream method for better coverage."""

    def test_stream_basic(self) -> None:
        """Basic streaming call works."""
        orchestrator = BlendOrchestrator()
        orchestrator.scorer = MagicMock()
        orchestrator.scorer.score.return_value = MagicMock(
            total=3,
            tier="LOW",
            task_type="general",
            intent_breakdown={},
        )

        orchestrator.compression = MagicMock()
        orchestrator.compression.compress.return_value = MagicMock(
            compressed="test",
            original_length=10,
            compressed_length=4,
        )

        orchestrator.strategy_gen = MagicMock()

        orchestrator.executor = MagicMock()
        # stream yields content strings that orchestrator wraps in dicts
        orchestrator.executor.stream.return_value = iter(["Hello", " world"])

        orchestrator.verifier = MagicMock()
        orchestrator.verifier.verify.return_value = MagicMock(passed=True, issues=[])

        orchestrator.resource_model = MagicMock()

        chunks = list(orchestrator.stream("Test prompt"))
        assert len(chunks) >= 2

    def test_stream_high_complexity_generates_strategy(self) -> None:
        """High complexity triggers strategy generation."""
        orchestrator = BlendOrchestrator()
        orchestrator.scorer = MagicMock()
        orchestrator.scorer.score.return_value = MagicMock(
            total=8,
            tier="HIGH",
            task_type="general",
            intent_breakdown={},
        )

        orchestrator.compression = MagicMock()
        orchestrator.compression.compress.return_value = MagicMock(
            compressed="test",
            original_length=10,
            compressed_length=4,
        )

        orchestrator.strategy = MagicMock()
        mock_output = MagicMock()
        mock_output.plan = ["Step 1", "Step 2"]
        mock_output.estimated_tokens = 50
        orchestrator.strategy.generate.return_value = mock_output

        orchestrator.executor = MagicMock()
        orchestrator.executor.stream.return_value = iter(["Result"])

        orchestrator.verifier = MagicMock()
        orchestrator.verifier.verify.return_value = MagicMock(passed=True, issues=[])

        orchestrator.resource_model = MagicMock()

        list(orchestrator.stream("Complex prompt"))
        # Verify strategy was generated
        orchestrator.strategy.generate.assert_called_once()
        # stream() passes through to executor without resource tracking

    def test_stream_with_long_prompt(self) -> None:
        """Long prompt may trigger compression."""
        orchestrator = BlendOrchestrator()
        orchestrator.scorer = MagicMock()
        orchestrator.scorer.score.return_value = MagicMock(
            total=5,
            tier="MEDIUM",
            task_type="general",
            intent_breakdown={},
        )

        orchestrator.executor = MagicMock()
        orchestrator.executor.stream.return_value = iter(["Result"])

        orchestrator.verifier = MagicMock()
        orchestrator.verifier.verify.return_value = MagicMock(passed=True, issues=[])

        orchestrator.resource_model = MagicMock()

        chunks = list(orchestrator.stream("Test prompt"))
        assert len(chunks) >= 1
