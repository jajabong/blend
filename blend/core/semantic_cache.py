"""L0 Semantic Cache - Embedding-based cache for high-frequency engineering tasks.

Catches high-frequency patterns (grep, test, commit, lint) and returns
cached results without invoking downstream model providers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    """A cached response entry."""

    response: str
    model_used: str
    tokens_saved: int
    hit_count: int = 1


@dataclass(frozen=True)
class CacheResult:
    """Result of a cache lookup."""

    hit: bool
    response: str | None = None
    model_used: str | None = None
    tokens_saved: int = 0
    reason: str = ""


HIGH_FREQ_PATTERNS = [
    # Git operations
    r"\b(commit|git add|git push|git pull|git checkout|git merge)\b",
    # Code inspection
    r"\b(grep|find|search|list|ls|dir)\b",
    # Testing
    r"\b(run test|pytest|npm test|jest|test|单元测试|集成测试)\b",
    # Linting
    r"\b(lint|eslint|prettier|format code|代码格式化)\b",
    # Build/compile
    r"\b(build|compile|make|install|npm install|pip install)\b",
    # Debug
    r"\b(debug|breakpoint|inspect|stack trace|日志)\b",
]


class SemanticCache:
    """Embedding-free semantic cache using pattern matching and prompt hashing.

    Uses a lightweight approach:
    - prompt_hash: SHA256 of lowercased prompt (fast, collision-resistant)
    - task_type: task category for cache partitioning
    - pattern signatures: high-freq engineering keywords
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._cache: dict[tuple[str, str], CacheEntry] = {}  # (prompt_hash, task_type) -> entry
        self._max_entries = max_entries

    def _compute_hash(self, prompt: str) -> str:
        """Compute SHA256 hash of lowercased prompt."""
        return hashlib.sha256(prompt.lower().encode()).hexdigest()[:32]

    def _get_pattern_signature(self, prompt: str) -> str:
        """Extract pattern signature from prompt for approximate matching."""
        sig_parts = []
        prompt_lower = prompt.lower()
        for pattern in HIGH_FREQ_PATTERNS:
            if re.search(pattern, prompt_lower):
                sig_parts.append(pattern)
        return "|".join(sorted(sig_parts)) if sig_parts else ""

    def _compute_similarity(self, prompt1: str, prompt2: str) -> float:
        """Compute lightweight similarity score (0.0 - 1.0)."""
        # Tokenize by whitespace
        tokens1 = set(prompt1.lower().split())
        tokens2 = set(prompt2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    def get(self, prompt: str, task_type: str = "general") -> CacheResult:
        """Look up cache for a prompt.

        Returns CacheResult with hit=True if found, hit=False otherwise.
        """
        prompt_hash = self._compute_hash(prompt)
        key = (prompt_hash, task_type)

        if key in self._cache:
            entry = self._cache[key]
            return CacheResult(
                hit=True,
                response=entry.response,
                model_used=entry.model_used,
                tokens_saved=entry.tokens_saved,
                reason="exact_hash_match",
            )

        # Check for pattern-based approximate hit (same task_type, high similarity)
        sig = self._get_pattern_signature(prompt)
        if sig:
            # Look for entries with same signature and high similarity
            for (cache_hash, cache_task), entry in self._cache.items():
                if cache_task == task_type:
                    cached_prompt = entry.response[:200]  # approximate
                    if self._compute_similarity(prompt, cached_prompt) > 0.85:
                        return CacheResult(
                            hit=True,
                            response=entry.response,
                            model_used=entry.model_used,
                            tokens_saved=entry.tokens_saved // 2,  # partial savings
                            reason="pattern_approximate_match",
                        )

        return CacheResult(hit=False, reason="cache_miss")

    def set(
        self,
        prompt: str,
        response: str,
        model_used: str,
        tokens_saved: int,
        task_type: str = "general",
    ) -> None:
        """Store a response in the cache."""
        if len(self._cache) >= self._max_entries:
            # Evict least recently used (simple: first item)
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        prompt_hash = self._compute_hash(prompt)
        key = (prompt_hash, task_type)
        self._cache[key] = CacheEntry(
            response=response,
            model_used=model_used,
            tokens_saved=tokens_saved,
        )

    def invalidate(self, prompt: str, task_type: str = "general") -> None:
        """Remove a specific entry from cache."""
        prompt_hash = self._compute_hash(prompt)
        key = (prompt_hash, task_type)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_hits = sum(e.hit_count for e in self._cache.values())
        total_tokens_saved = sum(e.tokens_saved for e in self._cache.values())
        return {
            "entries": len(self._cache),
            "total_hits": total_hits,
            "total_tokens_saved": total_tokens_saved,
        }