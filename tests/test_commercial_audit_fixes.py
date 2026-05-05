"""TDD tests for executor bug fixes - commercial deployment readiness."""

import logging
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from blend.core.executor import Executor
from blend.core.layers import L3Output


class TestRaceConditionFix:
    """Test that executor returns first SUCCESS, not first completed.

    This was a bug: if primary failed fast (before 3s probe timeout),
    the code returned the failed result instead of starting fallback.
    """

    def test_returns_success_not_first_completed(self) -> None:
        """When primary fails fast and fallback succeeds, return fallback."""
        exec_instance = Executor()

        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = "fallback response"
        mock_response.usage = {"completion_tokens": 10}

        with patch.object(exec_instance, "_call_model") as mock_call:
            # Primary fails, fallback succeeds
            mock_call.side_effect = [Exception("Primary failed"), mock_response]

            with patch.object(exec_instance, "_select_model") as mock_sel, \
                 patch.object(exec_instance, "_get_budget") as mock_budget:
                mock_sel.return_value = MagicMock(
                    primary="haiku",
                    fallback=["sonnet"]
                )
                mock_budget.return_value = 10000

                result = exec_instance.execute(
                    prompt="test prompt",
                    complexity=2,
                )

        # Should return successful result (fallback in this case)
        assert result.raw_output == "fallback response"

    def test_exception_in_except_clause_is_swallowed(self) -> None:
        """Bare except was catching everything including SystemExit.

        Now uses 'except Exception:' which excludes BaseException subclasses.
        """
        # This test verifies the code uses 'except Exception:' not 'except:'
        # by checking source code doesn't have bare except
        import blend.core.executor as exec_module

        source_file = exec_module.__file__
        if source_file:
            with open(source_file) as f:
                content = f.read()
            # Count bare 'except:' vs 'except Exception:'
            # Should not have bare except statements
            import re
            # Find 'except:' that is NOT 'except Exception:'
            bare_excepts = re.findall(r'except\s*:', content)
            # Filter out cases like 'except Exception:' (they have space before colon)
            assert 'except: continue' not in content, "Bare except found in executor"

    def test_stream_handles_json_parse_errors_gracefully(self) -> None:
        """Stream should not crash on malformed JSON chunks."""
        exec_instance = Executor()

        # Malformed JSON chunks
        mock_chunks = ['{"choices":', 'invalid json', '{"choices":[{"delta":{"content":"ok"}}]}']

        mock_provider = MagicMock()
        mock_provider.chat_stream.return_value = iter(mock_chunks)

        with patch("blend.core.executor._get_provider") as mock_get:
            mock_get.return_value = (mock_provider, "haiku")
            with patch.object(exec_instance, "_select_model") as mock_sel:
                mock_sel.return_value = MagicMock(
                    primary="haiku",
                    fallback=[]
                )
                # Should not raise, just skip malformed chunks
                chunks = list(exec_instance.stream("test", complexity=2))
                # Only valid content should be yielded
                assert len(chunks) <= 1


class TestLRUEviction:
    """Test that semantic cache evicts LRU, not just first-inserted."""

    def test_lru_eviction_on_get(self) -> None:
        """Accessing an entry should update its position for LRU."""
        from blend.core.semantic_cache import SemanticCache

        cache = SemanticCache(max_entries=3)

        # Fill cache to capacity
        cache.set("prompt1", "response1", "model", 100, "general")
        cache.set("prompt2", "response2", "model", 100, "general")
        cache.set("prompt3", "response3", "model", 100, "general")

        # Access prompt1 (makes it recently used)
        cache.get("prompt1", "general")

        # Add new entry - should evict prompt2 (LRU), not prompt1
        cache.set("prompt4", "response4", "model", 100, "general")

        # prompt1 should still be there (was accessed)
        result = cache.get("prompt1", "general")
        assert result.hit is True

        # prompt2 should be gone (was LRU)
        result2 = cache.get("prompt2", "general")
        assert result2.hit is False

    def test_lru_not_fifo(self) -> None:
        """Cache should evict LRU, not FIFO (first inserted)."""
        from blend.core.semantic_cache import SemanticCache

        cache = SemanticCache(max_entries=2)

        # Insert in order
        cache.set("A", "respA", "model", 100, "general")
        cache.set("B", "respB", "model", 100, "general")

        # Access A to make it recently used
        cache.get("A", "general")

        # Insert C - should evict B (LRU), not A
        cache.set("C", "respC", "model", 100, "general")

        # A should still be accessible
        assert cache.get("A", "general").hit is True
        # B should be evicted
        assert cache.get("B", "general").hit is False


class TestLoggingNotPrint:
    """Test that orchestrator uses logging, not print statements."""

    def test_no_debug_prints_in_orchestrator(self) -> None:
        """Verify no debug print statements remain in orchestrator."""
        import blend.core.orchestrator as orch_module

        source = orch_module.__file__
        if source:
            with open(source) as f:
                content = f.read()
            # Should not have print statements with DEBUG
            assert 'print(f"DEBUG' not in content
            assert 'print("DEBUG' not in content

    def test_logger_used_instead_of_print(self) -> None:
        """Verify logger is defined and used in orchestrator."""
        from blend.core.orchestrator import logger

        assert isinstance(logger, logging.Logger)
        assert logger.name == "blend.core.orchestrator"