"""L1 Compression utilities using Minimax."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionResult:
    """Result of prompt compression."""

    compressed: str
    compression_ratio: float
    original_length: int
    compressed_length: int


@dataclass(frozen=True)
class ConversationCompressionResult:
    """Result of conversation (history + current) compression."""

    current_compressed: str
    history_compressed: str | None
    total_compression_ratio: float
    history_preserved: bool


def compress_conversation(
    current_prompt: str,
    history: list[dict[str, str]] | None = None,
) -> ConversationCompressionResult:
    """Compress conversation with optional history.

    Args:
        current_prompt: The current user message
        history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]

    Returns:
        ConversationCompressionResult with compressed content
    """
    original_length = len(current_prompt)

    # Always compress current prompt
    current_result = compress_prompt(current_prompt)
    current_compressed = current_result.compressed

    # Process history if provided
    history_compressed: str | None = None
    history_preserved = False

    if history and len(history) > 0:
        # Keep latest 2 turns complete, compress older history
        if len(history) <= 2:
            # Short history - keep as-is
            history_preserved = True
            history_compressed = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history)
        else:
            # Long history - summarize older turns
            recent = history[-2:]
            older = history[:-2]
            history_compressed = _summarize_history(older) if older else ""
            # Append recent messages
            for msg in recent:
                history_compressed += f"\n{msg['role']}: {msg['content']}"
            history_preserved = True

    # Calculate total compression
    total_original = original_length + (
        sum(len(h.get("content", "")) for h in history) if history else 0
    )
    total_compressed = len(current_compressed) + (
        len(history_compressed) if history_compressed else 0
    )
    total_ratio = 1 - (total_compressed / total_original) if total_original > 0 else 0

    return ConversationCompressionResult(
        current_compressed=current_compressed,
        history_compressed=history_compressed,
        total_compression_ratio=total_ratio,
        history_preserved=history_preserved,
    )


def _summarize_history(messages: list[dict[str, str]]) -> str:
    """Summarize older conversation history."""
    if not messages:
        return ""

    # Simple extractive summary: combine first and last items
    summaries = []
    for msg in messages:
        content = msg.get("content", "")[:100]  # Truncate long messages
        summaries.append(f"[{msg.get('role', 'unknown')}]: {content}...")

    return "[Earlier conversation summarized]\n" + "\n".join(summaries[:4])


def compress_prompt(prompt: str, target_ratio: float = 0.6) -> CompressionResult:
    """Compress prompt using semantic compression via Minimax.

    Args:
        prompt: Original prompt to compress
        target_ratio: Target compression ratio (0.0-1.0)

    Returns:
        CompressionResult with compressed prompt and metrics
    """
    from blend.providers import MinimaxProvider

    original_length = len(prompt)

    provider = MinimaxProvider()
    messages = [
        {
            "role": "system",
            "content": f"""你是一个极致的语义压缩专家。
任务：将用户输入压缩到目标比例，保留100%核心需求。

压缩要求：
- 删除冗余表达、重复说明、口水话
- 保留核心指令、关键约束、输出格式要求
- 输出纯文本，无解释

输入：{prompt}

压缩输出：""",
        }
    ]

    try:
        response = provider.chat(messages=messages)
        compressed = response.content.strip()
    except Exception:
        # Fallback to simple compression if API fails
        compressed = _semantic_compress(prompt)

    compressed_length = len(compressed)
    compression_ratio = 1 - (compressed_length / original_length) if original_length > 0 else 0

    return CompressionResult(
        compressed=compressed,
        compression_ratio=compression_ratio,
        original_length=original_length,
        compressed_length=compressed_length,
    )


def _semantic_compress(prompt: str) -> str:
    """Apply simple semantic compression to prompt (fallback)."""
    compressed = " ".join(prompt.split())

    replacements = [
        ("please ", ""),
        ("could you ", ""),
        ("can you ", ""),
        ("would you mind ", ""),
        ("I would like you to ", "please "),
        ("Write a ", "Write "),
        ("Create a ", "Create "),
        ("Generate a ", "Generate "),
        ("Develop a ", "Develop "),
    ]

    for old, new in replacements:
        compressed = compressed.replace(old, new)

    return " ".join(compressed.split())
