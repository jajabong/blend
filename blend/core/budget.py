"""Resource Model - Advisor-Judge Quota Management for Mac mini M4.

三套资源核心规则：
1. MiniMax M2.7：每5小时重置4000次，单次上限4096Token
2. Gemini：15000次/无时间限制，单次最大64K Token
3. Claude：仅做催化剂，指导+终审，Token消耗最小化
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from time import time


class QuotaStatus(IntEnum):
    """Quota status levels."""

    HEALTHY = 0  # 额度充足
    WARNING = 1  # 额度不足20%
    CRITICAL = 2  # 额度耗尽/窗口超额
    WINDOW_EXCEEDED = 3  # 时间窗口超额


class ModelRole(IntEnum):
    """Model role in Advisor-Judge architecture."""

    EXECUTOR = 0  # Minimax: 基础执行层
    CATALYST = 1  # Gemini: 弹性催化剂
    ADVISOR_JUDGE = 2  # Claude: 催化剂+终审


@dataclass
class QuotaRule:
    """Quota rule for a model."""

    window_hours: float = 0  # 0 = 无时间窗口限制
    max_calls: int = 0  # 0 = 无调用次数限制
    token_per_call: int = 4096  # 单次目标Token数
    max_token_per_call: int = 0  # 0 = 无限制
    strategy: str = "fill_token"  # fill_token | batch_then_catalyst | advisor_judge_only


@dataclass
class CallRecord:
    """Record of a single API call."""

    model: str
    call_time: float
    tokens: int
    role: ModelRole


@dataclass
class WindowCounter:
    """Sliding window counter for rate limiting."""

    window_seconds: float
    max_calls: int
    calls: list[float] = field(default_factory=list)

    def can_call(self) -> bool:
        """Check if a new call is allowed within the window."""
        now = time()
        # Remove expired calls
        self.calls = [t for t in self.calls if now - t < self.window_seconds]
        return len(self.calls) < self.max_calls

    def record_call(self) -> bool:
        """Record a call. Returns True if allowed."""
        if not self.can_call():
            return False
        self.calls.append(time())
        return True

    def reset(self) -> None:
        """Reset the window."""
        self.calls.clear()

    @property
    def remaining(self) -> int:
        """Get remaining calls in current window."""
        now = time()
        self.calls = [t for t in self.calls if now - t < self.window_seconds]
        return max(0, self.max_calls - len(self.calls))

    @property
    def usage_percent(self) -> float:
        """Get usage percentage of current window."""
        if self.max_calls == 0:
            return 0.0
        now = time()
        self.calls = [t for t in self.calls if now - t < self.window_seconds]
        return (len(self.calls) / self.max_calls) * 100


class QuotaManager:
    """Manages quota for Advisor-Judge architecture.

    三层调度规则：
    - EXECUTOR (Minimax): 每5h窗口4000次，每次用足Token
    - CATALYST (Gemini): 15000次总上限，单次吃满Token
    - ADVISOR_JUDGE (Claude): 授信额度，仅做催化剂
    """

    # 按你的三套规则配置
    QUOTA_RULES: dict[str, QuotaRule] = {
        "minimax": QuotaRule(
            window_hours=5,
            max_calls=4000,
            token_per_call=4096,
            max_token_per_call=4096,
            strategy="fill_token",
        ),
        "gemini": QuotaRule(
            window_hours=0,  # 无时间限制
            max_calls=15000,  # 总15000次
            token_per_call=64000,  # 单次最大64K
            max_token_per_call=64000,
            strategy="batch_then_catalyst",
        ),
        "claude": QuotaRule(
            window_hours=0,
            max_calls=0,  # 按授信额度，不限次数
            token_per_call=1024,  # 催化剂模式，限制输出
            max_token_per_call=2048,
            strategy="advisor_judge_only",
        ),
    }

    # 模型角色映射
    MODEL_ROLES: dict[str, ModelRole] = {
        "minimax": ModelRole.EXECUTOR,
        "gemini_flash": ModelRole.CATALYST,
        "gemini_pro": ModelRole.CATALYST,
        "gemini_pro_ultra": ModelRole.CATALYST,
        "gemini_image_flash": ModelRole.CATALYST,
        "gemini_image_pro": ModelRole.CATALYST,
        "claude_sonnet": ModelRole.ADVISOR_JUDGE,
        "claude_opus": ModelRole.ADVISOR_JUDGE,
        "claude_haiku": ModelRole.ADVISOR_JUDGE,
    }

    def __init__(self) -> None:
        """Initialize quota manager with sliding window counters."""
        self._window_counters: dict[str, WindowCounter] = {}
        self._total_calls: dict[str, int] = {}  # 总调用次数（Gemini用）
        self._call_history: list[CallRecord] = []
        self._last_reset: dict[str, float] = {}

        # 初始化滑动窗口
        for name, rule in self.QUOTA_RULES.items():
            if rule.window_hours > 0:
                self._window_counters[name] = WindowCounter(
                    window_seconds=rule.window_hours * 3600,
                    max_calls=rule.max_calls,
                )
            if rule.max_calls > 0:
                self._total_calls[name] = 0
            self._last_reset[name] = time()

    def can_call(self, model: str) -> bool:
        """Check if a call is allowed for the model."""
        model_base = model.split("_")[0] if "_" in model else model

        # 检查滑动窗口
        if model_base in self._window_counters:
            return self._window_counters[model_base].can_call()

        # 检查总调用次数
        if model_base in self._total_calls:
            return (
                self._total_calls[model_base]
                < self.QUOTA_RULES.get(model_base, QuotaRule()).max_calls
            )

        return True

    def record_call(self, model: str, tokens: int = 0) -> bool:
        """Record a call and check if it was allowed."""
        model_base = model.split("_")[0] if "_" in model else model
        role = self.MODEL_ROLES.get(model, ModelRole.EXECUTOR)

        # 记录调用
        record = CallRecord(
            model=model,
            call_time=time(),
            tokens=tokens,
            role=role,
        )
        self._call_history.append(record)

        # 更新滑动窗口
        if model_base in self._window_counters:
            return self._window_counters[model_base].record_call()

        # 更新总调用次数
        if model_base in self._total_calls:
            self._total_calls[model_base] += 1
            return True

        return True

    def get_quota_status(self, model: str) -> QuotaStatus:
        """Get current quota status for a model."""
        model_base = model.split("_")[0] if "_" in model else model

        # 检查滑动窗口
        if model_base in self._window_counters:
            counter = self._window_counters[model_base]
            if counter.remaining <= 0:
                return QuotaStatus.CRITICAL
            if counter.usage_percent >= 80:
                return QuotaStatus.WARNING
            return QuotaStatus.HEALTHY

        # 检查总调用次数
        if model_base in self._total_calls:
            rule = self.QUOTA_RULES.get(model_base, QuotaRule())
            if rule.max_calls > 0:
                used = self._total_calls[model_base]
                remaining = rule.max_calls - used
                if remaining <= 0:
                    return QuotaStatus.CRITICAL
                if remaining < rule.max_calls * 0.2:
                    return QuotaStatus.WARNING
            return QuotaStatus.HEALTHY

        return QuotaStatus.HEALTHY

    def get_remaining_calls(self, model: str) -> int:
        """Get remaining calls for the model."""
        model_base = model.split("_")[0] if "_" in model else model

        if model_base in self._window_counters:
            return self._window_counters[model_base].remaining

        if model_base in self._total_calls:
            rule = self.QUOTA_RULES.get(model_base, QuotaRule())
            return max(0, rule.max_calls - self._total_calls[model_base])

        return -1  # 无限制

    def get_token_target(self, model: str) -> int:
        """Get target token count for a model (for Token filling)."""
        rule = self.QUOTA_RULES.get(model, QuotaRule())
        return rule.token_per_call

    def should_fill_token(self, model: str) -> bool:
        """Check if Token filling is enabled for this model."""
        rule = self.QUOTA_RULES.get(model, QuotaRule())
        return rule.strategy == "fill_token"

    def is_advisor_only(self, model: str) -> bool:
        """Check if this model should only be used as advisor/judge."""
        rule = self.QUOTA_RULES.get(model, QuotaRule())
        return rule.strategy == "advisor_judge_only"

    def is_catalyst(self, model: str) -> bool:
        """Check if this model should be used as catalyst (batch mode)."""
        rule = self.QUOTA_RULES.get(model, QuotaRule())
        return rule.strategy == "batch_then_catalyst"

    def get_model_role(self, model: str) -> ModelRole:
        """Get the role of a model in the architecture."""
        return self.MODEL_ROLES.get(model, ModelRole.EXECUTOR)

    def get_quota_summary(self) -> dict[str, dict[str, object]]:
        """Get quota summary for all models."""
        summary = {}
        for model in ["minimax", "gemini", "claude"]:
            rule = self.QUOTA_RULES.get(model, QuotaRule())
            remaining = self.get_remaining_calls(model)
            status = self.get_quota_status(model)

            summary[model] = {
                "status": status.name,
                "remaining_calls": remaining,
                "token_target": rule.token_per_call,
                "strategy": rule.strategy,
                "role": ModelRole(self.MODEL_ROLES.get(model, 0)).name,
            }
        return summary

    def reset_window(self, model: str) -> None:
        """Manually reset a model's sliding window."""
        if model in self._window_counters:
            self._window_counters[model].reset()
            self._last_reset[model] = time()

    def reset_all(self) -> None:
        """Reset all quotas (monthly reset)."""
        for counter in self._window_counters.values():
            counter.reset()
        self._total_calls = dict.fromkeys(self._total_calls, 0)
        self._call_history.clear()


# 全局实例
_quota_manager: QuotaManager | None = None


def get_quota_manager() -> QuotaManager:
    """Get the global quota manager instance."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager


# 保留旧接口兼容
class ResourceModel:
    """Legacy ResourceModel - now wraps QuotaManager."""

    BUDGETS = {
        "minimax": 100_000_000,
        "haiku": 100_000_000,
        "sonnet": 1_000_000,
        "opus": 500_000,
        "gemini": 200_000,
    }

    COSTS = {
        "minimax": 20,
        "haiku": 15,
        "sonnet": 0,
        "opus": 688,
        "gemini": 100,
    }

    def __init__(self) -> None:
        self._quota = get_quota_manager()

    def get_budget(self, model: str) -> int:
        return self.BUDGETS.get(model, 0)

    def track_consumption(self, model: str, tokens: int) -> None:
        self._quota.record_call(model, tokens)

    def get_status(self, model: str) -> dict[str, object]:
        status = self._quota.get_quota_status(model)
        return {
            "name": model,
            "remaining": self._quota.get_remaining_calls(model),
            "status": status.name,
        }

    def get_remaining(self, model: str) -> int:
        return self._quota.get_remaining_calls(model)

    def should_degrade(self, model: str) -> bool:
        return self._quota.get_quota_status(model) == QuotaStatus.CRITICAL

    def get_degraded_model(self, model: str) -> str:
        return "minimax"  # 降级到最便宜的模型

    def estimate_monthly_cost(self) -> float:
        return sum(self.COSTS.values())

    def reset_all(self) -> None:
        self._quota.reset_all()
