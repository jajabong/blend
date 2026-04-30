"""Tests for L4 compression layer."""



class TestCompressionTrigger:
    """Test CompressionTrigger.should_compress logic."""

    def test_below_threshold_returns_false(self) -> None:
        """Token count at or below threshold should not trigger compression."""
        from blend.core.compression import CompressionTrigger

        trigger = CompressionTrigger(threshold=200)
        assert trigger.should_compress(150) is False
        assert trigger.should_compress(200) is False

    def test_above_threshold_returns_true(self) -> None:
        """Token count above threshold should trigger compression."""
        from blend.core.compression import CompressionTrigger

        trigger = CompressionTrigger(threshold=200)
        assert trigger.should_compress(201) is True

    def test_agent_mode_skips_compression(self) -> None:
        """agent_mode=True should always return False regardless of token count."""
        from blend.core.compression import CompressionTrigger

        trigger = CompressionTrigger(threshold=200)
        # Even huge token count should not trigger when agent_mode=True
        assert trigger.should_compress(10000, agent_mode=True) is False
        assert trigger.should_compress(200, agent_mode=True) is False
        assert trigger.should_compress(0, agent_mode=True) is False

    def test_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        from blend.core.compression import CompressionTrigger

        trigger = CompressionTrigger(threshold=100)
        assert trigger.should_compress(50, agent_mode=False) is False
        assert trigger.should_compress(150, agent_mode=False) is True
