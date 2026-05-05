"""Tests for Token Estimation accuracy - Issue: len(text) // 4 is inaccurate."""

import pytest
from blend.core.executor import Executor


class TestTokenEstimation:
    """Test token estimation accuracy."""

    def test_short_text_oversestimate(self):
        """Short text (1-10 tokens) is significantly overestimated by // 4."""
        executor = Executor()

        test_cases = [
            ("Hi", 2),      # Actual: ~2 tokens, Estimate: 2//4=0
            ("Hello world", 2),  # Actual: ~2 tokens, Estimate: 2
            ("The", 1),     # Actual: ~1 token, Estimate: 0
            ("a", 1),       # Actual: ~1 token, Estimate: 0
        ]

        for text, actual in test_cases:
            estimate = executor._estimate_tokens(text)
            # For very short text, //4 severely underestimates
            print(f"Text: '{text}', Actual: {actual}, Estimate: {estimate}, Ratio: {estimate/actual if actual else 'N/A'}")

    def test_chinese_text_underestimate(self):
        """Chinese text uses more tokens per character than //4 assumes."""
        executor = Executor()

        # Chinese: each character is typically 1-2 tokens, not 0.25
        chinese_text = "你好世界"  # 4 chars, ~8 tokens
        estimate = executor._estimate_tokens(chinese_text)

        print(f"Chinese text: '{chinese_text}', Estimate: {estimate}, Actual chars: {len(chinese_text)}")
        # //4 gives 1, but actual is ~8 tokens
        # This is a significant underestimate

    def test_code_tokens_differ_from_prose(self):
        """Code tokens differ from prose - brackets, keywords, etc."""
        executor = Executor()

        code = "def foo(): return 42"  # ~6 tokens
        prose = "def foo return number"  # ~6 tokens

        code_estimate = executor._estimate_tokens(code)
        prose_estimate = executor._estimate_tokens(prose)

        # Same char count but different token counts
        print(f"Code estimate: {code_estimate}, Prose estimate: {prose_estimate}")

    def test_common_estimates(self):
        """Show estimation error for common prompt sizes."""
        executor = Executor()

        test_cases = [
            ("Write a function", 3),           # ~3 tokens
            ("Write a Python function that adds two numbers and returns the result", 15),  # ~15 tokens
            ("Write a Python function that:\n1. Takes a list of numbers\n2. Returns the sum", 20),  # ~20 tokens
        ]

        for text, expected_approx in test_cases:
            estimate = executor._estimate_tokens(text)
            error = abs(estimate - expected_approx) / expected_approx * 100
            print(f"Text: '{text[:30]}...', Expected: {expected_approx}, Estimate: {estimate}, Error: {error:.0f}%")


class TestTokenUsageExtraction:
    """Test that actual token usage is extracted from provider responses."""

    def test_extract_usage_from_response(self):
        """Test extracting token usage from LLMResponse."""
        from blend.providers.base import LLMResponse

        response = LLMResponse(
            content="Test response",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            raw={}
        )

        executor = Executor()
        usage = executor._extract_usage(response)

        # Should extract completion_tokens or total_tokens
        assert usage in [20, 30], f"Expected 20 or 30, got {usage}"

    def test_extract_usage_from_response_with_only_total(self):
        """Test when only total_tokens is available."""
        from blend.providers.base import LLMResponse

        response = LLMResponse(
            content="Test",
            model="test",
            usage={"total_tokens": 50},
            raw={}
        )

        executor = Executor()
        usage = executor._extract_usage(response)

        assert usage == 50

    def test_extract_usage_from_response_no_usage(self):
        """Test when no usage info available."""
        from blend.providers.base import LLMResponse

        response = LLMResponse(
            content="Test",
            model="test",
            usage={},
            raw={}
        )

        executor = Executor()
        usage = executor._extract_usage(response)

        assert usage is None