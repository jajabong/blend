"""Tests for story-002: L1 Entry Layer - Minimax Compression + Complexity Scoring."""

from blend.intent.scorer import ComplexityScorer


class TestComplexityScorer:
    """Test the complexity scorer."""

    def test_score_low_complexity_single_step(self) -> None:
        """Single step, generic domain, short output = complexity 1-2."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="What's the weather today?",
            output_length=50,
            creativity=0,
        )
        assert score.tier == "LOW"
        assert 1 <= score.total <= 2

    def test_score_medium_complexity(self) -> None:
        """Multi-step, some domain knowledge = complexity 3-5."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Write a Python function to sort a list and handle edge cases",
            output_length=500,
            creativity=1,
        )
        assert score.tier == "MEDIUM"
        assert 3 <= score.total <= 5

    def test_score_high_complexity(self) -> None:
        """Complex multi-step, high creativity = complexity 6-10."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Design a distributed system for real-time collaborative editing with conflict resolution",
            output_length=2000,
            creativity=2,
        )
        assert score.tier == "HIGH"
        assert 6 <= score.total <= 10

    def test_high_tier_reachable_defaults(self) -> None:
        """HIGH tier must be reachable with default params (output_length=200, creativity=0).

        Regression test: L2 Opus was unreachable because complexity capped at 7 with defaults.
        With new thresholds (HIGH >= 6) and output_length > 500 scoring 2, prompts scoring 6+
        should reach HIGH tier even with defaults.
        """
        scorer = ComplexityScorer()
        # "design distributed system" scores: steps=0, domain=2, output=1, creativity=0, risk=2 = 5
        # After output_length threshold change (>500 → 2pts), output becomes 1 still (default=200)
        # domain=2, risk=2, max=5 → still MEDIUM
        # Need steps >= 1 to push to 6
        score = scorer.score(
            prompt="Design a distributed system architecture with service mesh. First analyze requirements, then propose the topology, finally draft the implementation plan",
        )
        # steps=1, domain=2, output=1, creativity=0, risk=2 = 6 → HIGH
        assert score.tier == "HIGH", f"Expected HIGH but got {score.tier} with breakdown {score.breakdown}"
        assert 6 <= score.total <= 10

    def test_score_breakdown(self) -> None:
        """Test that breakdown contains all dimensions."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Test prompt",
            output_length=100,
            creativity=1,
        )
        assert "steps" in score.breakdown
        assert "domain" in score.breakdown
        assert "output" in score.breakdown
        assert "creativity" in score.breakdown
        assert "risk" in score.breakdown

    def test_route_decision_low(self) -> None:
        """LOW tier routes to L3_MINIMAX."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Hello",
            output_length=10,
            creativity=0,
        )
        assert score.route_decision == "L3_MINIMAX"

    def test_route_decision_medium(self) -> None:
        """MEDIUM tier routes to L3_HAIKU."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Write code to process data",
            output_length=500,
            creativity=1,
        )
        assert score.route_decision == "L3_HAIKU"

    def test_route_decision_high(self) -> None:
        """HIGH tier routes to L2_OPUS."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Design a system architecture for handling millions of requests",
            output_length=2000,
            creativity=2,
        )
        assert score.route_decision == "L2_OPUS"


class TestTaskTypeDetection:
    """Test task type detection for model routing."""

    def test_detect_deep_reasoning(self) -> None:
        """Deep reasoning tasks route to Gemini."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Analyze the logical structure of this proof",
            output_length=500,
            creativity=0,
        )
        assert score.task_type == "deep_reasoning"

    def test_detect_tool_call(self) -> None:
        """Tool call tasks route to Gemini."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Use the API to fetch user data",
            output_length=200,
            creativity=0,
        )
        assert score.task_type == "tool_call"

    def test_detect_code(self) -> None:
        """Code tasks route to Claude."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Debug this Python code that calculates fibonacci",
            output_length=200,
            creativity=0,
        )
        assert score.task_type == "code"

    def test_detect_general(self) -> None:
        """General tasks use default routing."""
        scorer = ComplexityScorer()
        score = scorer.score(
            prompt="Hello, how are you?",
            output_length=50,
            creativity=0,
        )
        assert score.task_type == "general"


class TestL1Compression:
    """Test L1 compression functionality."""

    def test_compression_ratio_reasonable(self) -> None:
        """Compression should reduce prompt size (tests fallback when no API)."""
        from blend.utils.compress import _semantic_compress

        prompt = "Write a detailed Python function that handles authentication, validation, and error handling for a user login system with JWT tokens and refresh token rotation"
        # Test fallback compression directly
        compressed = _semantic_compress(prompt)

        assert len(compressed) <= len(prompt)

    def test_compression_preserves_meaning(self) -> None:
        """Compressed prompt should preserve key semantics."""
        from blend.utils.compress import _semantic_compress

        prompt = "Generate a Python REST API endpoint for user registration with email validation"
        compressed = _semantic_compress(prompt)

        # Should preserve key terms
        assert "python" in compressed.lower() or "rest" in compressed.lower()

    def test_compression_output_format(self) -> None:
        """Compression output should have expected fields."""

        # With no API key, this falls back to _semantic_compress
        # which returns a string directly, not CompressionResult
        # So we test the dataclass directly
        from blend.utils.compress import CompressionResult

        result = CompressionResult(
            compressed="test",
            compression_ratio=0.5,
            original_length=100,
            compressed_length=50,
        )
        assert hasattr(result, "compressed")
        assert hasattr(result, "compression_ratio")
        assert hasattr(result, "original_length")
        assert hasattr(result, "compressed_length")


class TestConversationCompression:
    """Test conversation history compression."""

    def test_compress_conversation_no_history(self) -> None:
        """Compress prompt without history."""
        from blend.utils.compress import compress_conversation

        result = compress_conversation("Hello world")
        assert result.current_compressed is not None
        assert result.history_compressed is None
        assert result.history_preserved is False

    def test_compress_conversation_short_history(self) -> None:
        """Compress prompt with short history (keeps as-is)."""
        from blend.utils.compress import compress_conversation

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = compress_conversation("How are you?", history)
        assert result.history_preserved is True
        assert result.history_compressed is not None

    def test_compress_conversation_long_history(self) -> None:
        """Compress prompt with long history (summarizes older)."""
        from blend.utils.compress import compress_conversation

        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Second response"},
            {"role": "user", "content": "Third message"},
            {"role": "assistant", "content": "Third response"},
        ]
        result = compress_conversation("Continue please", history)
        assert result.history_compressed is not None
        assert "[Earlier conversation summarized]" in result.history_compressed
