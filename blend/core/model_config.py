"""Model configuration loader from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model."""

    provider: str
    model_name: str
    tier: str
    cost_per_1k: float
    budget_monthly: int | None
    fallback: list[str] = field(default_factory=list)
    l2_token_limit: int | None = None
    min_batch_threshold: float | None = None


@dataclass(frozen=True)
class ComplexityRouting:
    """Routing config for a complexity range."""

    max_complexity: int = 0  # Maximum complexity for this tier (from YAML)
    primary: str = ""
    description: str = ""


@dataclass(frozen=True)
class ResourceWarnings:
    """Resource warning thresholds."""

    claude_usage_threshold: float
    gemini_min_threshold: float
    sonnet_budget: int
    haiku_budget: int
    minimax_budget: int


@dataclass(frozen=True)
class DegradationRule:
    """Degradation rule from one model to another."""

    from_model: str
    to_model: str


@dataclass(frozen=True)
class QuotaRule:
    """Quota rule for a model (Advisor-Judge architecture)."""

    window_hours: float = 0
    max_calls: int = 0
    token_per_call: int = 4096
    max_token_per_call: int = 0
    strategy: str = "fill_token"
    role: str = "executor"


@dataclass(frozen=True)
class ModelRegistry:
    """Registry of all model configurations."""

    tiers: dict[str, ModelConfig]
    complexity_routing: dict[str, ComplexityRouting]
    gemini_task_types: list[str]
    resource_warnings: ResourceWarnings
    degradation_rules: list[DegradationRule]
    quota_rules: dict[str, QuotaRule] = field(default_factory=dict)


def _find_models_yaml() -> Path:
    """Find models.yaml relative to this file."""
    # Try current directory first (for CLI usage)
    local = Path("models.yaml")
    if local.exists():
        return local

    # Try relative to this module (for package usage)
    module_dir = Path(__file__).parent.parent
    pkg = module_dir / "models.yaml"
    if pkg.exists():
        return pkg

    # Try home directory
    home = Path.home() / "blend" / "models.yaml"
    if home.exists():
        return home

    raise FileNotFoundError(
        "models.yaml not found. Searched: . , ~/.blend/models.yaml, package directory",
    )


@lru_cache(maxsize=1)
def load_model_registry() -> ModelRegistry:
    """Load model registry from YAML (cached)."""
    yaml_path = _find_models_yaml()

    with open(yaml_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Parse tiers
    tiers: dict[str, ModelConfig] = {}
    for name, cfg in raw.get("tiers", {}).items():
        tiers[name] = ModelConfig(
            provider=cfg["provider"],
            model_name=cfg["model_name"],
            tier=cfg["tier"],
            cost_per_1k=cfg["cost_per_1k"],
            budget_monthly=cfg.get("budget_monthly"),
            fallback=cfg.get("fallback", []),
            l2_token_limit=cfg.get("l2_token_limit"),
            min_batch_threshold=cfg.get("min_batch_threshold"),
        )

    # Parse complexity routing
    routing: dict[str, ComplexityRouting] = {}
    for key, cfg in raw.get("complexity_routing", {}).items():
        routing[key] = ComplexityRouting(
            max_complexity=cfg.get("max_complexity", 0),
            primary=cfg["primary"],
            description=cfg.get("description", ""),
        )

    # Parse resource warnings
    rw_cfg = raw.get("resource_warnings", {})
    rw = ResourceWarnings(
        claude_usage_threshold=rw_cfg.get("claude_usage_threshold", 0.9),
        gemini_min_threshold=rw_cfg.get("gemini_min_threshold", 0.5),
        sonnet_budget=rw_cfg.get("sonnet_budget", 100),
        haiku_budget=rw_cfg.get("haiku_budget", 100),
        minimax_budget=rw_cfg.get("minimax_budget", 50),
    )

    # Parse degradation rules
    deg_rules: list[DegradationRule] = []
    for rule in raw.get("degradation", {}).get("rules", []):
        deg_rules.append(
            DegradationRule(from_model=rule["from"], to_model=rule["to"]),
        )

    # Parse quota rules (Advisor-Judge)
    quota_rules: dict[str, QuotaRule] = {}
    for key, cfg in raw.get("quota_rules", {}).items():
        quota_rules[key] = QuotaRule(
            window_hours=cfg.get("window_hours", 0),
            max_calls=cfg.get("max_calls", 0),
            token_per_call=cfg.get("token_per_call", 4096),
            max_token_per_call=cfg.get("max_token_per_call", 0),
            strategy=cfg.get("strategy", "fill_token"),
            role=cfg.get("role", "executor"),
        )

    return ModelRegistry(
        tiers=tiers,
        complexity_routing=routing,
        gemini_task_types=raw.get("gemini_task_types", []),
        resource_warnings=rw,
        degradation_rules=deg_rules,
        quota_rules=quota_rules,
    )


# Backwards compatibility helpers
def get_model_map() -> dict[str, tuple[str, str]]:
    """Get MODEL_MAP for executor (provider class name, model name)."""
    registry = load_model_registry()
    return {
        name: (cfg.provider, cfg.model_name)
        for name, cfg in registry.tiers.items()
    }


def get_fallback_chain() -> dict[str, list[str]]:
    """Get FALLBACK_CHAIN for executor."""
    registry = load_model_registry()
    return {
        name: cfg.fallback
        for name, cfg in registry.tiers.items()
    }


def get_model_cost() -> dict[str, float]:
    """Get MODEL_COST per 1K tokens."""
    registry = load_model_registry()
    return {
        name: cfg.cost_per_1k
        for name, cfg in registry.tiers.items()
    }


def get_gemini_task_types() -> set[str]:
    """Get Gemini task types set."""
    registry = load_model_registry()
    return set(registry.gemini_task_types)


def get_quota_rules() -> dict[str, QuotaRule]:
    """Get quota rules for Advisor-Judge architecture."""
    registry = load_model_registry()
    return registry.quota_rules


def get_model_role(model: str) -> str:
    """Get model role (executor/catalyst/advisor_judge)."""
    registry = load_model_registry()
    tier = registry.tiers.get(model)
    if tier and hasattr(tier, "role"):
        return tier.role  # type: ignore
    # 从quota_rules中获取
    quota = registry.quota_rules.get(model.replace("claude_", "claude_").replace("gemini_", "gemini_"))
    if quota:
        return quota.role
    return "executor"


def get_advisor_judge_models() -> list[str]:
    """Get list of models that should only be used as advisor/judge."""
    return ["claude_sonnet", "claude_opus", "claude_haiku"]


def get_catalyst_models() -> list[str]:
    """Get list of models that should be used as catalyst."""
    return ["gemini_flash", "gemini_pro", "gemini_pro_ultra", "gemini_image_flash", "gemini_image_pro"]


def get_complexity_thresholds() -> dict[str, int]:
    """Get complexity thresholds from models.yaml.

    Returns:
        dict with low_max and medium_max complexity values.
        These are loaded from complexity_routing config.

    Raises:
        FileNotFoundError: if models.yaml is not found.

    """
    registry = load_model_registry()
    routing = registry.complexity_routing

    low_cfg = routing.get("low")
    medium_cfg = routing.get("medium")

    # Default values matching legacy hardcoded behavior
    low_max = low_cfg.max_complexity if low_cfg else 2
    medium_max = medium_cfg.max_complexity if medium_cfg else 5

    return {
        "low_max": low_max,
        "medium_max": medium_max,
    }


def is_low_complexity(complexity: int) -> bool:
    """Check if complexity is LOW tier."""
    thresholds = get_complexity_thresholds()
    return complexity <= thresholds["low_max"]


def is_medium_complexity(complexity: int) -> bool:
    """Check if complexity is MEDIUM tier."""
    thresholds = get_complexity_thresholds()
    return thresholds["low_max"] < complexity <= thresholds["medium_max"]


def is_high_complexity(complexity: int) -> bool:
    """Check if complexity is HIGH tier."""
    thresholds = get_complexity_thresholds()
    return complexity > thresholds["medium_max"]


