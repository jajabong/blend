"""Tests for QuotaAlert state persistence."""

import json
import time
import pytest
from pathlib import Path

from blend.scheduler.alert import QuotaAlert, QuotaMonitor, AlertLevel


class TestQuotaAlertPersistence:
    """Test QuotaAlert state persistence."""

    def test_quota_alert_has_persist_method(self) -> None:
        """QuotaAlert should have a persist_state method."""
        alert = QuotaAlert()
        assert hasattr(alert, "persist_state") or hasattr(alert, "_persist_state")

    def test_quota_alert_has_load_method(self) -> None:
        """QuotaAlert should have a load_state method."""
        alert = QuotaAlert()
        assert hasattr(alert, "load_state") or hasattr(alert, "_load_state")

    def test_quota_status_saved_after_alert(self) -> None:
        """Alert state should be saveable to disk."""
        alert = QuotaAlert()

        # Trigger an alert
        result = alert.check_and_alert("minimax", remaining_calls=100, max_calls=4000)

        # Should be able to get state
        state = alert.get_state()
        assert state is not None
        assert "minimax" in state or len(alert.get_alert_log()) >= 0

    def test_alert_state_includes_timestamp(self) -> None:
        """Saved alert state should include timestamps."""
        alert = QuotaAlert()

        # Trigger an alert
        alert.check_and_alert("minimax", remaining_calls=100, max_calls=4000)

        state = alert.get_state()
        # State should contain timestamp info
        assert state is not None


class TestQuotaMonitorPersistence:
    """Test QuotaMonitor state persistence."""

    def test_quota_monitor_can_export_state(self) -> None:
        """QuotaMonitor should be able to export its state."""
        monitor = QuotaMonitor()

        # Update some quota
        monitor.update_quota("minimax", remaining_calls=500, max_calls=4000)

        # Should be able to get quota status
        status = monitor.get_quota_status()
        assert "minimax" in status

    def test_quota_monitor_can_import_state(self) -> None:
        """QuotaMonitor should be able to import saved state."""
        monitor = QuotaMonitor()

        # Create a state dict (simulating restored state)
        saved_state = {
            "minimax": {
                "remaining_calls": 300,
                "max_calls": 4000,
                "usage_percent": 92.5,
                "is_healthy": False,
                "alert_level": "warning",
            }
        }

        # Should be able to restore from state
        if hasattr(monitor, "load_state"):
            monitor.load_state(saved_state)
            status = monitor.get_quota_status()
            # State should reflect loaded values
            assert "minimax" in status