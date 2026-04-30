
from unittest.mock import MagicMock, patch

import pytest

from blend.core.executor import Executor
from tests.utils.provider_simulator import ProviderSimulator


def test_executor_resilience_with_simulator():
    """Verify that the Executor correctly handles simulated provider failures."""
    simulator = ProviderSimulator()
    simulator.set_failure(Exception("Network error"))

    mock_provider = MagicMock()
    mock_provider.chat.side_effect = lambda *args, **kwargs: simulator.simulate(lambda: "success")

    with patch("blend.core.executor._get_provider", return_value=(mock_provider, "model")):
        executor = Executor()

        # Expectation: Executor should attempt to fallback or return error
        with pytest.raises(Exception, match="Network error"):
            executor.execute("test prompt", complexity=1)
