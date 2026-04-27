"""Tests for story-005: L4 Compression Layer - Minimax Secondary Compression."""

from blend.core.compression import CompressionTrigger, L4Compressor


class TestCompressionTrigger:
    """Test compression triggering logic."""

    def test_trigger_above_threshold(self) -> None:
        """Should trigger when tokens > 200."""
        trigger = CompressionTrigger()
        assert trigger.should_compress(201) is True

    def test_no_trigger_at_threshold(self) -> None:
        """Should not trigger at exactly 200 tokens."""
        trigger = CompressionTrigger()
        assert trigger.should_compress(200) is False

    def test_no_trigger_below_threshold(self) -> None:
        """Should not trigger when tokens < 200."""
        trigger = CompressionTrigger()
        assert trigger.should_compress(150) is False

    def test_threshold_configurable(self) -> None:
        """Threshold should be configurable."""
        trigger = CompressionTrigger(threshold=100)
        assert trigger.should_compress(101) is True
        assert trigger.should_compress(99) is False


class TestL4Compressor:
    """Test L4 compression functionality."""

    def test_compress_output_format(self) -> None:
        """Compression should return L4Output."""
        compressor = L4Compressor()
        result = compressor.compress(
            text="This is a test response that needs compression",
            original_tokens=100,
        )
        assert hasattr(result, "compressed_output")
        assert hasattr(result, "original_tokens")
        assert hasattr(result, "compressed_tokens")
        assert hasattr(result, "compression_ratio")

    def test_compression_ratio_calculation(self) -> None:
        """Compression ratio should be calculated correctly."""
        compressor = L4Compressor()
        result = compressor.compress(
            text="A" * 1000,  # 1000 char input
            original_tokens=1000,
        )
        assert 0 <= result.compression_ratio <= 1

    def test_compression_preserves_content(self) -> None:
        """Compressed output should contain meaningful content."""
        compressor = L4Compressor()
        result = compressor.compress(
            text="The quick brown fox jumps over the lazy dog",
            original_tokens=100,
        )
        # Should preserve key content, not be empty
        assert len(result.compressed_output) > 0
        assert result.compressed_tokens < result.original_tokens

    def test_minimum_compression_ratio(self) -> None:
        """Compression ratio should be at least 0 (no negative)."""
        compressor = L4Compressor()
        result = compressor.compress(
            text="Short text",
            original_tokens=10,
        )
        assert result.compression_ratio >= 0

    def test_full_output_format(self) -> None:
        """Should return complete L4Output."""
        from blend.core.layers import L4Output

        compressor = L4Compressor()
        result = compressor.compress(
            text="Test response",
            original_tokens=50,
        )
        # Result should be compatible with L4Output
        output = L4Output(
            compressed_output=result.compressed_output,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_ratio=result.compression_ratio,
        )
        assert isinstance(output, L4Output)
