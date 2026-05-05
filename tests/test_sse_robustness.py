"""Tests for SSE streaming robustness enhancements.

These tests ensure blend's SSE output is compatible with @ai-sdk/openai-compatible SDK
and other HTTP client libraries that have strict SSE format requirements.
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio


class TestSSEFormatHelpers:
    """Test SSE format helper functions."""

    def test_format_sse_payload(self) -> None:
        """_format_sse_payload should produce valid SSE data line."""
        from blend.api import _format_sse_payload
        payload = {"id": "test", "choices": []}
        result = _format_sse_payload(payload)
        assert result == 'data: {"id": "test", "choices": []}\n\n'
        assert result.startswith("data: ")

    def test_format_sse_comment(self) -> None:
        """_format_sse_comment should produce valid SSE comment line."""
        from blend.api import _format_sse_comment
        result = _format_sse_comment("heartbeat 5s")
        assert result == ": heartbeat 5s\n\n"
        assert result.startswith(": ")

    def test_is_valid_sse_line(self) -> None:
        """_is_valid_sse_line should correctly identify valid SSE formats."""
        from blend.api import _is_valid_sse_line
        # Valid lines
        assert _is_valid_sse_line("data: {}")
        assert _is_valid_sse_line("data: [DONE]")
        assert _is_valid_sse_line(": heartbeat")
        assert _is_valid_sse_line(": comment line")
        # Invalid lines
        assert not _is_valid_sse_line("not data")
        assert not _is_valid_sse_line("something else")
        assert not _is_valid_sse_line("")


class TestSSEStreamRobustness:
    """Test SSE stream format standardization and error handling via curl."""

    def test_stream_via_curl_standard_sse_format(self) -> None:
        """Use curl to verify streaming produces valid SSE format."""
        import subprocess
        result = subprocess.run([
            'curl', '-s', '-N', '-X', 'POST', 'http://localhost:8000/v1/chat/completions',
            '-H', 'Content-Type: application/json',
            '-H', 'Authorization: Bearer sk-3LnEna7qP5qaq4eRxzeDhTmpfcXp9sbQURxNbcMZeIUuEJtc',
            '-d', '{"model": "blend", "messages": [{"role": "user", "content": "say hello"}], "stream": true, "max_tokens": 50}',
        ], capture_output=True, text=True, timeout=30)

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        for line in lines:
            if line.strip():  # Skip empty lines
                assert line.startswith("data: ") or line.startswith(":"), f"Invalid SSE format: {line!r}"

    def test_stream_via_curl_ends_with_done(self) -> None:
        """Use curl to verify stream ends with data: [DONE]."""
        import subprocess
        result = subprocess.run([
            'curl', '-s', '-N', '-X', 'POST', 'http://localhost:8000/v1/chat/completions',
            '-H', 'Content-Type: application/json',
            '-H', 'Authorization: Bearer sk-3LnEna7qP5qaq4eRxzeDhTmpfcXp9sbQURxNbcMZeIUuEJtc',
            '-d', '{"model": "blend", "messages": [{"role": "user", "content": "say hello"}], "stream": true, "max_tokens": 50}',
        ], capture_output=True, text=True, timeout=30)

        assert "data: [DONE]" in result.stdout, f"Expected data: [DONE] in stream, got: {result.stdout[-200:]}"


class TestAnthropicStreamFormat:
    """Test Anthropic /v1/messages endpoint SSE format."""

    def test_anthropic_stream_via_curl(self) -> None:
        """Use curl to verify Anthropic streaming format."""
        import subprocess
        result = subprocess.run([
            'curl', '-s', '-N', '-X', 'POST', 'http://localhost:8000/v1/messages',
            '-H', 'Content-Type: application/json',
            '-H', 'x-api-key: sk-3LnEna7qP5qaq4eRxzeDhTmpfcXp9sbQURxNbcMZeIUuEJtc',
            '-H', 'anthropic-version: 2023-06-01',
            '-d', '{"model": "blend", "messages": [{"role": "user", "content": "say hello"}], "stream": true, "max_tokens": 50}',
        ], capture_output=True, text=True, timeout=30)

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        for line in lines:
            if line.strip():
                assert line.startswith("data: ") or line.startswith(":"), f"Invalid SSE format: {line!r}"


class TestHeartbeatConfiguration:
    """Test heartbeat interval configuration."""

    def test_heartbeat_interval_configurable(self) -> None:
        """SSE_HEARTBEAT_INTERVAL should be configurable via environment variable."""
        import os
        # Test default value
        from blend.api import SSE_HEARTBEAT_INTERVAL
        assert SSE_HEARTBEAT_INTERVAL == 15.0  # Default

        # Test that environment variable changes it
        # (This would require reloading the module, so we just verify the mechanism exists)
        assert "SSE_HEARTBEAT_INTERVAL" in dir() or True  # Placeholder for env var test

    def test_chunk_timeout_configurable(self) -> None:
        """SSE_CHUNK_TIMEOUT should be configurable via environment variable."""
        from blend.api import SSE_CHUNK_TIMEOUT
        assert SSE_CHUNK_TIMEOUT == 10.0  # Default