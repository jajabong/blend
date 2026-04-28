"""Tests for L1 removal - verify no dead code remains."""

from blend.core.orchestrator import BlendOrchestrator


def test_smart_compress_always_returns_false() -> None:
    """_smart_compress should always return False per v1.7 L1 removal."""
    orchestrator = BlendOrchestrator()
    should_compress, result = orchestrator._smart_compress(
        prompt="This is a test prompt", complexity=5
    )
    assert should_compress is False, "_smart_compress should always return False after L1 removal"
    assert result is None, "_smart_compress should always return None after L1 removal"


def test_smart_compress_ignores_parameters() -> None:
    """_smart_compress should ignore prompt and complexity parameters."""
    orchestrator = BlendOrchestrator()

    # Various prompt lengths
    short_prompt = "Hi"
    long_prompt = "Explain " * 1000

    # Various complexities
    for complexity in range(1, 11):
        s1, r1 = orchestrator._smart_compress(short_prompt, complexity)
        s2, r2 = orchestrator._smart_compress(long_prompt, complexity)
        assert s1 is False and r1 is None
        assert s2 is False and r2 is None
