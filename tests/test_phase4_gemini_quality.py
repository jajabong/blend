"""Tests for Phase 4: L5 Gemini Quality Assessment (Lightweight).

TDD Phase:
- RED: Write failing tests for Gemini-driven quality gate
- GREEN: Implement QualityVerifier.gemini_evaluate() and integration
- IMPROVE: Verify all pass
"""

from unittest.mock import MagicMock, patch

from blend.core.verifier import QualityVerifier


class MockGeminiResponse:
    """Mock Gemini response."""

    def __init__(self, content: str) -> None:
        self.content = content


class TestGeminiQualityGateEnabled:
    """Test that HIGH quality_level triggers Gemini evaluation."""

    def test_high_quality_calls_gemini_evaluate(self) -> None:
        """verify(HIGH) should call gemini_evaluate internally."""
        verifier = QualityVerifier()

        with patch.object(verifier, "gemini_evaluate") as mock_gemini:
            mock_gemini.return_value = {
                "passed": True,
                "relevance": 90,
                "accuracy": 85,
                "completeness": 80,
                "issues": [],
            }

            result = verifier.verify(
                output="A comprehensive response about distributed systems.",
                quality_level="HIGH",
                layer_path="L1>L2>L3>L4>L5",
                gemini_used=False,
                gemini_context_percent=0,
                output_tokens=100,
            )

            mock_gemini.assert_called_once()
            assert result.passed is True

    def test_high_quality_gemini_fail_rejects(self) -> None:
        """Gemini evaluation fail should cause overall rejection."""
        verifier = QualityVerifier()

        with patch.object(verifier, "gemini_evaluate") as mock_gemini:
            mock_gemini.return_value = {
                "passed": False,
                "relevance": 40,
                "accuracy": 30,
                "completeness": 20,
                "issues": ["Irrelevant to prompt", "Inaccurate claim"],
            }

            result = verifier.verify(
                output="Wrong answer unrelated to the task.",
                quality_level="HIGH",
                layer_path="L1>L2>L3>L4>L5",
                gemini_used=False,
                gemini_context_percent=0,
                output_tokens=100,
            )

            assert result.passed is False
            assert "quality" in result.rejection_reason.lower() or "gemini" in result.rejection_reason.lower()

    def test_low_quality_skips_gemini(self) -> None:
        """verify(LOW/MEDIUM) should NOT call Gemini."""
        verifier = QualityVerifier()

        with patch.object(verifier, "gemini_evaluate") as mock_gemini:
            result = verifier.verify(
                output="A short answer.",
                quality_level="LOW",
                layer_path="L1>L3>L5",
                gemini_used=False,
                gemini_context_percent=0,
                output_tokens=50,
            )

            mock_gemini.assert_not_called()
            # Should still pass based on rule gates
            assert isinstance(result.passed, bool)

    def test_medium_quality_skips_gemini(self) -> None:
        """verify(MEDIUM) should NOT call Gemini."""
        verifier = QualityVerifier()

        with patch.object(verifier, "gemini_evaluate") as mock_gemini:
            verifier.verify(
                output="A medium response about the topic.",
                quality_level="MEDIUM",
                layer_path="L1>L3>L5",
                gemini_used=False,
                gemini_context_percent=0,
                output_tokens=200,
            )

            mock_gemini.assert_not_called()


class TestGeminiEvaluateMethod:
    """Test gemini_evaluate method."""

    def test_gemini_evaluate_returns_scores(self) -> None:
        """gemini_evaluate returns structured quality scores."""
        verifier = QualityVerifier()
        mock_response = MockGeminiResponse(
            '{"passed": true, "relevance": 95, "accuracy": 90, "completeness": 88, "issues": []}'
        )

        with patch("blend.providers.lemonapi.LemonProvider") as mock_provider:
            instance = MagicMock()
            instance.chat.return_value = mock_response
            mock_provider.return_value = instance

            scores = verifier.gemini_evaluate(
                output="A high-quality response about distributed systems.",
                quality_level="HIGH",
            )

            assert scores["passed"] is True
            assert scores["relevance"] == 95
            assert scores["accuracy"] == 90
            assert scores["completeness"] == 88
            assert scores["issues"] == []

    def test_gemini_evaluate_parses_issues(self) -> None:
        """gemini_evaluate parses issues from Gemini response."""
        verifier = QualityVerifier()
        mock_response = MockGeminiResponse(
            '{"passed": false, "relevance": 40, "accuracy": 35, "completeness": 30, '
            '"issues": ["Irrelevant content", "Missing key details"]}'
        )

        with patch("blend.providers.lemonapi.LemonProvider") as mock_provider:
            instance = MagicMock()
            instance.chat.return_value = mock_response
            mock_provider.return_value = instance

            scores = verifier.gemini_evaluate(
                output="Wrong answer completely off-topic.",
                quality_level="HIGH",
            )

            assert scores["passed"] is False
            assert len(scores["issues"]) == 2

    def test_gemini_evaluate_fallback_on_error(self) -> None:
        """Gemini API failure returns safe default (pass with low scores)."""
        verifier = QualityVerifier()

        with patch("blend.providers.lemonapi.LemonProvider") as mock_provider:
            instance = MagicMock()
            instance.chat.side_effect = Exception("API error")
            mock_provider.return_value = instance

            scores = verifier.gemini_evaluate(
                output="Some output.",
                quality_level="HIGH",
            )

            # Should return safe default, not raise
            assert scores["passed"] is True
            assert scores["relevance"] == 0
            assert scores["accuracy"] == 0
            assert scores["completeness"] == 0
            assert "API error" in scores["issues"][0]


class TestGeminiGateIntegration:
    """Test that Gemini gate is merged into overall gates dict."""

    def test_gemini_gate_added_to_gates(self) -> None:
        """verify() includes Gemini gate in gates_checked."""
        verifier = QualityVerifier()

        with patch.object(verifier, "gemini_evaluate") as mock_gemini:
            mock_gemini.return_value = {
                "passed": True,
                "relevance": 90,
                "accuracy": 85,
                "completeness": 80,
                "issues": [],
            }

            result = verifier.verify(
                output="High quality output.",
                quality_level="HIGH",
                layer_path="L1>L2>L3>L5",
                gemini_used=False,
                gemini_context_percent=0,
                output_tokens=100,
            )

            assert "gemini_quality" in result.gates_checked
            assert result.gates_checked["gemini_quality"] is True

    def test_gemini_gate_fail_propagates(self) -> None:
        """gemini_quality gate fail causes overall fail."""
        verifier = QualityVerifier()

        with patch.object(verifier, "gemini_evaluate") as mock_gemini:
            mock_gemini.return_value = {
                "passed": False,
                "relevance": 30,
                "accuracy": 25,
                "completeness": 20,
                "issues": ["Very low quality"],
            }

            result = verifier.verify(
                output="Bad output.",
                quality_level="HIGH",
                layer_path="L1>L2>L3>L5",
                gemini_used=False,
                gemini_context_percent=0,
                output_tokens=100,
            )

            assert result.gates_checked["gemini_quality"] is False
            assert result.passed is False
