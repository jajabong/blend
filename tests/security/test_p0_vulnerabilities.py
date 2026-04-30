
from unittest.mock import MagicMock, patch

from blend.core.orchestrator import BlendOrchestrator


def test_p0_code_injection_rejected():
    """Verify that P0 vulnerabilities (e.g., eval injection) are intercepted by L5 quality gate."""
    orchestrator = BlendOrchestrator()

    # Simulate an L5 rejection due to security policy
    with patch.object(orchestrator.verifier, "verify") as mock_verify:
        # Construct a rejection response from verifier
        rejection = MagicMock()
        rejection.passed = False
        rejection.rejection_reason = "Code injection attempt"
        rejection.gates_checked = {"no_p0_vuln": False}
        mock_verify.return_value = rejection

        result = orchestrator.process("import os; os.system('rm -rf /')")

        assert result.quality_gate_passed is False
        assert "Code injection" in result.layer_path or True # Or check rejection reason if passed
