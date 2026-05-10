"""Context Budget Manager - Prevents provider context overflow during multi-step tool loops."""

from typing import Any


def estimate_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    """Estimate token count for a message list.

    Uses rough heuristic: ~4 chars per token for mixed content.
    Reserves 20%% of context window for response (handled by caller).

    Args:
        messages: List of message dicts

    Returns:
        Estimated token count

    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multimodal: estimate text parts, count media as fixed overhead
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total += len(part.get("text", "")) // 4
                    else:
                        total += 50  # media overhead
        elif isinstance(content, str):
            total += len(content) // 4
        # Tool role messages with tool_call_id have minimal overhead
        if msg.get("tool_call_id"):
            total += 20
    return total


def truncate_messages(
    messages: list[dict[str, Any]],
    max_tokens: int,
    reserve_tokens: int = 100,
) -> list[dict[str, Any]]:
    """Truncate oldest non-system messages to fit within token budget.

    Preserves the most recent messages up to max_tokens.

    Args:
        messages: Full message history
        max_tokens: Maximum tokens to keep (soft limit before reserve)
        reserve_tokens: Reserve for response buffer (default 100)

    Returns:
        Truncated message list preserving recent messages

    """
    if not messages:
        return []

    effective_max = max(50, max_tokens - reserve_tokens)
    current_tokens = estimate_tokens_from_messages(messages)

    if current_tokens <= effective_max:
        return list(messages)

    result: list[dict[str, Any]] = []
    used_tokens = 0

    # Walk backwards through all messages, keeping recent ones
    # This naturally drops oldest messages first
    for msg in reversed(messages):
        msg_tokens = _estimate_single_message_tokens(msg)
        if used_tokens + msg_tokens <= effective_max:
            result.append(msg)
            used_tokens += msg_tokens
        else:
            break

    # Reverse to restore chronological order
    result.reverse()
    return result


def _estimate_single_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate tokens for a single message."""
    content = msg.get("content", "")
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    total += len(part.get("text", "")) // 4
                else:
                    total += 50
        return total
    if isinstance(content, str):
        return len(content) // 4
    return 10


def check_context_budget(
    messages: list[dict[str, Any]],
    context_limit: int = 128000,
    usage_percent: float = 0.80,
) -> bool:
    """Check if messages are within safe context budget.

    Args:
        messages: Current message history
        context_limit: Provider context window (default 128k for Claude)
        usage_percent: Max usage threshold (default 80%%)

    Returns:
        True if within budget, False if truncation recommended

    """
    tokens = estimate_tokens_from_messages(messages)
    return tokens <= int(context_limit * usage_percent)
