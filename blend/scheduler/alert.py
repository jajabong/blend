"""Quota Alert - Real-time Monitoring and Alerting.

额度告警模块：
1. 实时监控三模型额度状态
2. 低于20%告警 (WARNING)
3. 低于5%严重告警 (CRITICAL)
4. 额度耗尽时触发通知 (EXHAUSTED)
5. 支持回调通知机制
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AlertLevel(Enum):
    """Alert severity levels."""

    WARNING = "warning"  # 额度低于20%
    CRITICAL = "critical"  # 额度低于5%
    EXHAUSTED = "exhausted"  # 额度耗尽


@dataclass
class Alert:
    """An alert instance."""

    model: str
    level: AlertLevel
    message: str
    remaining_calls: int
    usage_percent: float
    timestamp: float | None = None


@dataclass
class QuotaStatus:
    """Current quota status for a model."""

    model: str
    remaining_calls: int
    max_calls: int
    usage_percent: float
    is_healthy: bool
    alert_level: AlertLevel | None


class QuotaMonitor:
    """Real-time quota monitoring with alerting.

    监控三模型额度，低于阈值时触发告警。
    """

    # 告警阈值
    WARNING_THRESHOLD = 20.0  # 低于20%告警
    CRITICAL_THRESHOLD = 5.0  # 低于5%严重告警

    def __init__(self, on_alert_callback: Callable[[Alert], None] | None = None) -> None:
        """初始化监控器。

        Args:
            on_alert_callback: 告警回调函数，签名为 (Alert) -> None

        """
        self._callback = on_alert_callback
        self._active_alerts: dict[str, Alert] = {}  # model -> Alert
        self._alert_history: list[Alert] = []
        self._quota_status: dict[str, QuotaStatus] = {}

    def update_quota(self, model: str, remaining_calls: int, max_calls: int) -> None:
        """更新模型额度状态。

        Args:
            model: 模型名称
            remaining_calls: 剩余调用次数
            max_calls: 最大调用次数

        """
        usage_percent = (
            ((max_calls - remaining_calls) / max_calls * 100) if max_calls > 0 else 100.0
        )

        # 确定告警级别
        alert_level = None
        if remaining_calls <= 0:
            alert_level = AlertLevel.EXHAUSTED
        elif usage_percent >= 95:
            alert_level = AlertLevel.CRITICAL
        elif usage_percent >= 80:
            alert_level = AlertLevel.WARNING

        # 创建或更新状态
        self._quota_status[model] = QuotaStatus(
            model=model,
            remaining_calls=remaining_calls,
            max_calls=max_calls,
            usage_percent=usage_percent,
            is_healthy=alert_level is None,
            alert_level=alert_level,
        )

        # 处理告警
        if alert_level is not None:
            self._trigger_alert(model, alert_level, remaining_calls, usage_percent)
        elif model in self._active_alerts:
            # 额度恢复，清除告警
            del self._active_alerts[model]

    def _trigger_alert(self, model: str, level: AlertLevel, remaining: int, usage: float) -> None:
        """触发告警。"""
        # 检查是否已有同级或更高级别告警
        existing = self._active_alerts.get(model)
        if existing and existing.level.value <= level.value:
            # 更新已有告警但不再触发回调
            existing.remaining_calls = remaining
            existing.usage_percent = usage
            return

        # 创建新告警
        message = self._build_message(model, level, remaining)
        alert = Alert(
            model=model,
            level=level,
            message=message,
            remaining_calls=remaining,
            usage_percent=usage,
        )

        self._active_alerts[model] = alert
        self._alert_history.append(alert)

        # 触发回调
        if self._callback:
            self._callback(alert)

    def _build_message(self, model: str, level: AlertLevel, remaining: int) -> str:
        """构建告警消息。"""
        if level == AlertLevel.EXHAUSTED:
            return f"{model} 额度已耗尽！剩余调用: {remaining}"
        if level == AlertLevel.CRITICAL:
            return f"{model} 额度严重不足！剩余: {remaining} 次"
        return f"{model} 额度低于20%，剩余: {remaining} 次"

    def get_active_alerts(self) -> list[Alert]:
        """获取当前活跃告警列表。"""
        return list(self._active_alerts.values())

    def get_alert_history(self) -> list[Alert]:
        """获取告警历史。"""
        return list(self._alert_history)

    def is_healthy(self, model: str) -> bool:
        """检查模型是否健康（无告警）。"""
        return model not in self._active_alerts

    def clear_alerts(self, model: str) -> None:
        """清除指定模型的告警。"""
        if model in self._active_alerts:
            del self._active_alerts[model]
            # 更新状态为健康
            if model in self._quota_status:
                self._quota_status[model].is_healthy = True
                self._quota_status[model].alert_level = None

    def clear_all_alerts(self) -> None:
        """清除所有告警。"""
        self._active_alerts.clear()
        for status in self._quota_status.values():
            status.is_healthy = True
            status.alert_level = None

    def get_quota_status(self) -> dict[str, dict[str, object]]:
        """获取所有模型的额度状态。"""
        return {
            model: {
                "remaining_calls": s.remaining_calls,
                "max_calls": s.max_calls,
                "usage_percent": s.usage_percent,
                "is_healthy": s.is_healthy,
                "alert_level": s.alert_level.value if s.alert_level else None,
            }
            for model, s in self._quota_status.items()
        }


class QuotaAlert:
    """Standalone quota alerting utility.

    独立告警工具，可单独使用或与QuotaMonitor配合。
    支持状态持久化到磁盘，重启后可恢复告警状态。
    """

    DEFAULT_STATE_FILE = ".blend_quota_alert.json"

    def __init__(self, state_file: str | None = None) -> None:
        self._monitor = QuotaMonitor()
        self._alert_log: list[Alert] = []
        self._state_file = state_file or self.DEFAULT_STATE_FILE

    def check_and_alert(self, model: str, remaining_calls: int, max_calls: int) -> Alert | None:
        """检查额度并触发告警。

        Args:
            model: 模型名称
            remaining_calls: 剩余调用次数
            max_calls: 最大调用次数

        Returns:
            如果触发告警返回Alert，否则返回None

        """
        self._monitor.update_quota(model, remaining_calls, max_calls)
        alerts = self._monitor.get_active_alerts()
        for alert in alerts:
            if alert.model == model:
                self._alert_log.append(alert)
                return alert
        return None

    def get_active_alerts(self) -> list[Alert]:
        """获取当前告警。"""
        return self._monitor.get_active_alerts()

    def get_alert_log(self) -> list[Alert]:
        """获取告警日志。"""
        return list(self._alert_log)

    def is_healthy(self, model: str) -> bool:
        """检查模型是否健康。"""
        return self._monitor.is_healthy(model)

    def get_state(self) -> dict[str, object]:
        """获取告警状态用于持久化。

        Returns:
            dict包含alert_log和quota_status
        """
        return {
            "alert_log": [
                {
                    "model": a.model,
                    "level": a.level.value,
                    "message": a.message,
                    "remaining_calls": a.remaining_calls,
                    "usage_percent": a.usage_percent,
                    "timestamp": a.timestamp or time.time(),
                }
                for a in self._alert_log
            ],
            "quota_status": self._monitor.get_quota_status(),
        }

    def persist_state(self, path: str | None = None) -> str:
        """持久化告警状态到文件。

        Args:
            path: 文件路径，默认使用state_file

        Returns:
            保存的文件路径
        """
        state_file = path or self._state_file
        state = self.get_state()
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        return state_file

    def load_state(self, state: dict[str, object] | None = None, path: str | None = None) -> None:
        """从文件或字典加载告警状态。

        Args:
            state: 可选的dict状态，直接加载
            path: 文件路径，从文件加载
        """
        if state is None:
            state_file = path or self._state_file
            if Path(state_file).exists():
                with open(state_file, encoding="utf-8") as f:
                    state = json.load(f)

        if state is None:
            return

        # Restore alert log
        self._alert_log = []
        for entry in state.get("alert_log", []):
            self._alert_log.append(Alert(
                model=entry["model"],
                level=AlertLevel(entry["level"]),
                message=entry["message"],
                remaining_calls=entry["remaining_calls"],
                usage_percent=entry["usage_percent"],
                timestamp=entry.get("timestamp"),
            ))

        # Restore quota status
        quota_status = state.get("quota_status", {})
        for model, status in quota_status.items():
            if isinstance(status, dict):
                self._monitor.update_quota(
                    model,
                    status.get("remaining_calls", 0),
                    status.get("max_calls", 1),
                )


# 全局实例
_alert: QuotaAlert | None = None


def get_quota_alert() -> QuotaAlert:
    """获取全局告警实例。"""
    global _alert
    if _alert is None:
        _alert = QuotaAlert()
    return _alert
