"""L4 Compression Layer - Minimax Secondary Compression for Large Outputs."""

from dataclasses import dataclass

from blend.providers import MinimaxProvider


@dataclass(frozen=True)
class CompressionResult:
    """Result of L4 compression."""

    compressed_output: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float


class CompressionTrigger:
    """Determines when L4 compression should be applied."""

    def __init__(self, threshold: int = 1000) -> None:
        """Initialize with compression threshold.

        Args:
            threshold: Token count above which compression triggers (default 1000,
                aligned with L5 Gate 6 threshold).
        """
        self.threshold = threshold

    def should_compress(self, token_count: int, agent_mode: bool = False) -> bool:
        """Check if compression should be applied.

        Args:
            token_count: Number of tokens in output
            agent_mode: If True, skip compression to preserve tool result fidelity

        Returns:
            True if compression should be applied
        """
        if agent_mode:
            return False
        return token_count > self.threshold


# Prompt template for L4 compression via Minimax
L4_COMPRESSION_PROMPT = """你是一个文本压缩工具。请将用户提供的文本压缩至原始长度的约50%，保留所有关键信息和核心含义。直接输出压缩结果，不要包含任何解释或说明。

要压缩的文本：
{text}

压缩结果："""


class L4Compressor:
    """Compresses L3 output when it exceeds token threshold via Minimax API."""

    def __init__(self, threshold: int = 200) -> None:
        """Initialize L4 compressor.

        Args:
            threshold: Token count above which compression triggers
        """
        self.trigger = CompressionTrigger(threshold)
        self._provider = MinimaxProvider()

    def compress(self, text: str, original_tokens: int) -> CompressionResult:
        """Compress text output via Minimax API.

        Args:
            text: Text to compress
            original_tokens: Token count of original text

        Returns:
            CompressionResult with compressed output
        """
        try:
            compressed = self._call_minimax_api(text)
            compressed_tokens = len(compressed) // 4
        except Exception:
            # Fallback to local compression if API call fails
            compressed = self._semantic_compress(text)
            compressed_tokens = original_tokens // 2

        # Guard: if compressed is longer than original, use semantic fallback ratio
        if compressed_tokens >= original_tokens:
            compressed_tokens = max(1, original_tokens // 2)

        compression_ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0

        return CompressionResult(
            compressed_output=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
        )

    def _call_minimax_api(self, text: str) -> str:
        """Call Minimax API for semantic compression."""
        prompt = L4_COMPRESSION_PROMPT.format(text=text[: 20_000])  # Guard: max 20k chars
        messages = [{"role": "user", "content": prompt}]
        response = self._provider.chat(messages=messages)
        return str(response.content).strip()

    def _semantic_compress(self, text: str) -> str:
        """Apply semantic compression to text."""
        compressed = " ".join(text.split())

        replacements = [
            (" in order to ", " to "),
            (" due to the fact that ", " because "),
            (" in the event that ", " if "),
            (" at this point in time ", " now "),
            (" for the purpose of ", " to "),
            (" with regard to ", " about "),
            (" in accordance with ", " per "),
            (" has the ability to ", " can "),
            (" take into consideration ", " consider "),
        ]

        for old, new in replacements:
            compressed = compressed.replace(old, new)

        filler_phrases = [
            "basically",
            "actually",
            "literally",
            "seriously",
            "obviously",
            "clearly",
            "simply",
        ]
        for phrase in filler_phrases:
            compressed = compressed.replace(f" {phrase} ", " ")

        compressed = " ".join(compressed.split())

        if len(compressed) < len(text) * 0.5:
            compressed = text[: len(text) // 2]

        return compressed
