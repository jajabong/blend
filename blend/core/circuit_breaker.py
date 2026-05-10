"""Circuit breaker for provider resilience with persistence and active probing."""

from __future__ import annotations

import json
import os
import threading
import time
from enum import Enum
from typing import Any


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocked
    HALF_OPEN = "half_open"  # Testing recovery


# Global disable flag - defaults to False (circuit breakers ENABLED)
DISABLE_CIRCUIT_BREAKER = os.environ.get("DISABLE_CIRCUIT_BREAKER", "false").lower() in ("true", "1", "yes")


class CircuitBreaker:
    """Circuit breaker with adaptive recovery and persistence support."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        base_recovery_timeout: float = 30.0,
        state_data: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.base_recovery_timeout = base_recovery_timeout

        # Initialize from state data if provided (persistence)
        if state_data:
            self._state = CircuitState(state_data.get("state", "closed"))
            self._failure_count = state_data.get("failure_count", 0)
            self._consecutive_trips = state_data.get("consecutive_trips", 0)
            self._last_failure_time = state_data.get("last_failure_time")
            self._lockout_duration = state_data.get("lockout_duration", base_recovery_timeout)
        else:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._consecutive_trips = 0
            self._last_failure_time = None
            self._lockout_duration = base_recovery_timeout

        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state with adaptive recovery window."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            # Monotonic time check
            if time.monotonic() - self._last_failure_time >= self._lockout_duration:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dict for persistence."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "consecutive_trips": self._consecutive_trips,
            "last_failure_time": self._last_failure_time,
            "lockout_duration": self._lockout_duration,
        }

    def allow_request(self) -> bool:
        """Check if request can proceed.

        Note: Caller must call record_success() or record_failure() after the probe,
        not this method. This method only checks if request can proceed.
        """
        # Disabled via environment variable - allow everything
        if DISABLE_CIRCUIT_BREAKER:
            return True
        with self._lock:
            current = self.state
            if current == CircuitState.CLOSED:
                return True
            if current == CircuitState.OPEN:
                return False
            # HALF_OPEN: allow one probe request through
            # State transition to CLOSED happens via record_success() on probe success,
            # or stays HALF_OPEN on probe failure (via record_failure)
            return True

    def record_success(self) -> None:
        """Record success and reset health."""
        with self._lock:
            self._failure_count = 0
            self._consecutive_trips = 0
            self._state = CircuitState.CLOSED
            self._last_failure_time = None
            self._lockout_duration = self.base_recovery_timeout

    def record_failure(self, error_code: int | None = None) -> None:
        """Record failure with error triage."""
        # Disabled via environment variable - don't record failures
        if DISABLE_CIRCUIT_BREAKER:
            return
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            # Debug
            # print(f"DEBUG: Breaker {self.name} record_failure {error_code}, count={self._failure_count}")

            # If we were probing (HALF_OPEN) and probe failed, go back to OPEN
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                # Reset failure count so next recovery cycle starts fresh
                self._failure_count = 0
                return

            # Error Triage logic
            if error_code == 401:
                # Unauthorized = Admin failure. Massive lockout (1 hour)
                self._lockout_duration = 3600
                self._state = CircuitState.OPEN
                self._consecutive_trips += 1
                return

            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._consecutive_trips += 1
                    # Exponential backoff for generic failures
                    self._lockout_duration = min(
                        self.base_recovery_timeout * (2 ** self._consecutive_trips),
                        3600 * 24, # Max 1 day
                    )
                self._state = CircuitState.OPEN


class CircuitBreakerRegistry:
    """Registry with persistence and background heartbeat."""

    HEALTH_FILE = ".blend_health.json"

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self._load_state()

        # Start background heartbeat thread
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread = threading.Thread(target=self._run_heartbeat, daemon=True)
        self._heartbeat_thread.start()

    def _load_state(self) -> None:
        """Load states from disk."""
        if os.path.exists(self.HEALTH_FILE):
            try:
                with open(self.HEALTH_FILE) as f:
                    data = json.load(f)
                    for name, state in data.items():
                        self._breakers[name] = CircuitBreaker(name=name, state_data=state)
            except Exception:
                pass

    def save_state(self) -> None:
        """Save current health states to disk."""
        with self._lock:
            data = {name: b.to_dict() for name, b in self._breakers.items()}
            try:
                with open(self.HEALTH_FILE, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass

    def get(self, name: str, **kwargs: Any) -> CircuitBreaker:
        """Get or create breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=kwargs.get("failure_threshold", 5),
                    base_recovery_timeout=kwargs.get("base_recovery_timeout", 30.0),
                )
            return self._breakers[name]

    def _run_heartbeat(self) -> None:
        """Background thread to periodically save state and check health."""
        while not self._stop_heartbeat.is_set():
            time.sleep(60) # Interval
            self.save_state()

    def reset_all(self) -> None:
        """Reset all health memories."""
        with self._lock:
            for cb in self._breakers.values():
                cb.record_success()
            if os.path.exists(self.HEALTH_FILE):
                try:
                    os.remove(self.HEALTH_FILE)
                except OSError:
                    pass


# Singleton Registry
_registry: CircuitBreakerRegistry | None = None

def get_registry() -> CircuitBreakerRegistry:
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry
