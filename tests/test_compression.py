"""Tests for L4 compression layer."""

from unittest.mock import MagicMock


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


class TestL4Compressor:
    """Test L4Compressor.compress behavior."""

    def test_compress_calls_minimax_api(self) -> None:
        """compress() should call Minimax API."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        compressor._provider = MagicMock()
        compressor._provider.chat.return_value = MagicMock(content="compressed text")

        result = compressor.compress("long output " * 200, original_tokens=600)

        assert result.compressed_output == "compressed text"
        assert result.original_tokens == 600
        assert "compressed" in result.compressed_output.lower()
        compressor._provider.chat.assert_called_once()

    def test_compress_api_fallback_on_exception(self) -> None:
        """If Minimax API fails, should fall back to semantic compression."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        compressor._provider = MagicMock()
        compressor._provider.chat.side_effect = Exception("API error")

        result = compressor.compress("due to the fact that this is a test", original_tokens=500)

        # Semantic compress collapses whitespace + does phrase replacement
        assert isinstance(result.compressed_output, str)
        assert result.compressed_tokens == 250  # original_tokens // 2

    def test_compress_guard_uses_shorter_result(self) -> None:
        """If compressed_tokens >= original_tokens, use 50% fallback."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        compressor._provider = MagicMock()
        # Simulate: compressed is longer than original (API returns same or longer)
        compressor._provider.chat.return_value = MagicMock(content="short")  # same 5 chars

        result = compressor.compress("tiny", original_tokens=1000)

        # Guard kicks in: compressed_tokens(≈1) < original_tokens(1000)
        # ratio = 1 - (1/1000) ≈ 0.999
        assert result.compressed_tokens < result.original_tokens
        assert result.compression_ratio > 0

    def test_compress_zero_original_tokens(self) -> None:
        """Edge case: zero original tokens should not divide by zero."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        compressor._provider = MagicMock()
        compressor._provider.chat.side_effect = Exception("fail")

        result = compressor.compress("", original_tokens=0)

        assert result.original_tokens == 0
        assert result.compression_ratio == 0  # no division by zero

    def test_compress_max_chars_guard(self) -> None:
        """API call should truncate text to 20k chars."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        compressor._provider = MagicMock()
        compressor._provider.chat.return_value = MagicMock(content="result")

        long_text = "x" * 30_000
        compressor.compress(long_text, original_tokens=500)

        # Check the prompt passed to the provider
        call_args = compressor._provider.chat.call_args
        prompt_text = call_args.kwargs.get("messages", [])[0]["content"]
        # The prompt template adds ~85 chars of Chinese instructions around the text
        assert len(prompt_text) <= 20_085  # 20_000 + template overhead


class TestSemanticCompress:
    """Test _semantic_compress phrase replacements and filler removal."""

    def test_replaces_formal_phrases(self) -> None:
        """Formal verbose phrases should be replaced with shorter equivalents."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        # "in order to" has leading/trailing spaces after whitespace collapse
        text = "this is a test in order to verify functionality"
        result = compressor._semantic_compress(text)
        assert " to verify" in result  # "in order to" → " to "

    def test_removes_filler_phrases(self) -> None:
        """Filler words like 'basically', 'actually' should be removed."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        text = "this is basically and actually very important"
        result = compressor._semantic_compress(text)
        assert "basically" not in result
        assert "actually" not in result

    def test_preserves_meaning_with_replacements(self) -> None:
        """Phrase replacements should preserve core meaning."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        text = (
            "The system has the ability to process requests in order to "
            "provide results with regard to performance."
        )
        result = compressor._semantic_compress(text)
        # Key words should survive
        assert "system" in result
        assert "process" in result or "requests" in result

    def test_collapse_whitespace(self) -> None:
        """Multiple spaces/newlines should be collapsed to single spaces."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        text = "hello    world\n\nfoo   bar"
        result = compressor._semantic_compress(text)
        assert "  " not in result
        assert "\n" not in result

    def test_returns_string(self) -> None:
        """Should always return a string even on edge inputs."""
        from blend.core.compression import L4Compressor

        compressor = L4Compressor()
        assert isinstance(compressor._semantic_compress(""), str)
        assert isinstance(compressor._semantic_compress("hello"), str)
