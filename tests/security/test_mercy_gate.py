
from unittest.mock import MagicMock, patch

from blend.core.orchestrator import BlendOrchestrator


def test_mercy_gate_retry_logic():
    """Verify that minor quality flaws trigger a retry loop."""
    orchestrator = BlendOrchestrator()

    with patch.object(orchestrator.verifier, "verify") as mock_verify, \
         patch.object(orchestrator.executor, "execute_messages") as mock_exec, \
         patch.object(orchestrator.executor, "execute") as mock_exec_initial:

        # 1. First execution
        mock_exec_initial.return_value = MagicMock(raw_output="Initial", tokens_used=10)

        # 2. Mock Verification: fail then pass
        fail_mock = MagicMock(passed=False, rejection_reason="Minor error")
        pass_mock = MagicMock(passed=True)
        side_effects = [fail_mock, pass_mock]
        mock_verify.side_effect = side_effects

        # 3. Mock Retry execution
        mock_exec.return_value = MagicMock(content="Fixed output", model_used="sonnet", tokens_used=10)

        result = orchestrator.process("A prompt")

        assert mock_verify.call_count >= 1
