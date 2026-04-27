"""Five-layer architecture for blend."""

from dataclasses import dataclass
from enum import StrEnum


class Layer(StrEnum):
    """Layer identifiers."""

    L1_ENTRY = "L1"  # Minimax compression + complexity scoring
    L2_STRATEGY = "L2"  # Opus strategy generation (HIGH only)
    L3_EXECUTE = "L3"  # Dynamic model selection
    L4_COMPRESS = "L4"  # Minimax secondary compression
    L5_VERIFY = "L5"  # Graded quality gate


@dataclass(frozen=True)
class L1Output:
    """Output from L1 entry layer."""

    compressed_prompt: str
    complexity_score: int
    complexity_breakdown: dict[str, int]
    route_decision: str
    l1_compressed: bool
    compression_ratio: float


@dataclass(frozen=True)
class L2Output:
    """Output from L2 strategy layer."""

    plan: list[str]
    quality_redlines: list[str]
    boundary_cases: list[str]
    model_hint: str
    estimated_tokens: int


@dataclass(frozen=True)
class L3Output:
    """Output from L3 execution layer."""

    raw_output: str
    model_used: str
    tokens_used: int
    tokens_budget_remaining: int
    quality_gate_passed: bool


@dataclass(frozen=True)
class L4Output:
    """Output from L4 compression layer."""

    compressed_output: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float


@dataclass(frozen=True)
class L5Output:
    """Output from L5 verification layer."""

    final_output: str
    quality_gate_passed: bool
    gates_checked: dict[str, bool]
    quality_level: str
    rejection_reason: str | None
