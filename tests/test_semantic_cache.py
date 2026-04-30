"""Tests for Semantic Cache (L0)."""

import pytest
from blend.core.semantic_cache import SemanticCache, CacheResult, CacheEntry, HIGH_FREQ_PATTERNS


class TestSemanticCache:
    """Test SemanticCache functionality."""

    def test_cache_initialization(self) -> None:
        """Cache initializes with empty state."""
        cache = SemanticCache(max_entries=100)
        stats = cache.stats()
        assert stats["entries"] == 0
        assert stats["total_hits"] == 0
        assert stats["total_tokens_saved"] == 0

    def test_cache_set_and_get_exact(self) -> None:
        """Exact prompt hash match returns cached response."""
        cache = SemanticCache()
        cache.set(
            prompt="run pytest on tests/",
            response="All tests passed",
            model_used="haiku",
            tokens_saved=150,
            task_type="code",
        )
        result = cache.get("run pytest on tests/", task_type="code")
        assert result.hit is True
        assert result.response == "All tests passed"
        assert result.model_used == "haiku"
        assert result.tokens_saved == 150
        assert result.reason == "exact_hash_match"

    def test_cache_miss_returns_false(self) -> None:
        """Non-existent prompt returns hit=False."""
        cache = SemanticCache()
        result = cache.get("nonexistent prompt xyz", task_type="general")
        assert result.hit is False
        assert result.response is None
        assert result.reason == "cache_miss"

    def test_cache_eviction_on_max_entries(self) -> None:
        """Cache evicts oldest entry when max_entries exceeded."""
        cache = SemanticCache(max_entries=2)
        cache.set("prompt1", "response1", "haiku", 100, "code")
        cache.set("prompt2", "response2", "sonnet", 200, "code")
        # Third entry should evict first
        cache.set("prompt3", "response3", "opus", 300, "code")

        # prompt1 should be evicted
        result = cache.get("prompt1", task_type="code")
        assert result.hit is False

        # prompt2 and prompt3 should still be there
        assert cache.get("prompt2", task_type="code").hit is True
        assert cache.get("prompt3", task_type="code").hit is True

    def test_cache_invalidate(self) -> None:
        """Invalidate removes specific entry."""
        cache = SemanticCache()
        cache.set("test prompt", "test response", "haiku", 50, "general")
        assert cache.get("test prompt", "general").hit is True

        cache.invalidate("test prompt", "general")
        assert cache.get("test prompt", "general").hit is False

    def test_cache_clear(self) -> None:
        """Clear removes all entries."""
        cache = SemanticCache()
        cache.set("p1", "r1", "haiku", 10, "code")
        cache.set("p2", "r2", "sonnet", 20, "code")
        assert cache.stats()["entries"] == 2

        cache.clear()
        assert cache.stats()["entries"] == 0

    def test_cache_stats_tracking(self) -> None:
        """Stats accurately track cache state."""
        cache = SemanticCache()
        cache.set("p1", "r1", "haiku", 100, "code")
        cache.set("p2", "r2", "sonnet", 200, "code")

        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["total_tokens_saved"] == 300


class TestSemanticCachePatternSignatures:
    """Test HIGH_FREQ_PATTERNS coverage for engineering tasks."""

    @pytest.mark.parametrize(
        "prompt,should_match",
        [
            ("run pytest on test_*.py", True),
            ("grep -r 'TODO' src/", True),
            ("git commit -m 'fix bug'", True),
            ("npm install express", True),
            ("build the project with maven", True),
            ("debug the stack trace", True),
            ("what is 2+2", False),  # general math
            ("write a poem about cats", False),  # creative
            ("explain quantum entanglement", False),  # explanation
        ],
    )
    def test_high_freq_patterns(self, prompt: str, should_match: bool) -> None:
        """High-frequency engineering patterns are correctly identified."""
        import re
        prompt_lower = prompt.lower()
        matched = any(re.search(p, prompt_lower) for p in HIGH_FREQ_PATTERNS)
        assert matched == should_match, f"Prompt '{prompt}' matched={matched}, expected={should_match}"