"""L4 Compression Layer - Retained for Compatibility.

DEPRECATED: L4Compressor removed in v2.0 (negative ROI: +18s latency for $0.003 savings).
Only CompressionTrigger and CompressionResult are still used by L1 pipeline.
"""

from dataclasses import dataclass


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
