"""Circuit breaker for provider resilience."""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import TypedDict


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Failing, requests are blocked immediately
    HALF_OPEN = "half_open"  # Testing, one request allowed through


class CircuitBreaker:
    """Circuit breaker that trips after consecutive failures, auto-recovers.

    States:
        CLOSED   → Normal: all requests pass. On failure, increment counter.
        OPEN     → Block all requests immediately. After recovery_timeout, move to HALF_OPEN.
        HALF_OPEN → Allow one request through. Success → CLOSED. Failure → OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "default",
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            name: Identifier for this circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state, auto-transitioning OPEN → HALF_OPEN on timeout."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request can proceed, False if circuit is open.
        """
        with self._lock:
            current = self.state
            if current == CircuitState.CLOSED:
                return True
            if current == CircuitState.OPEN:
                return False
            # HALF_OPEN: allow exactly one request through
            self._state = CircuitState.CLOSED
            return True

    def record_success(self) -> None:
        """Record a successful request. Resets circuit to CLOSED."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._last_failure_time = None

    def record_failure(self) -> None:
        """Record a failed request. Opens circuit after threshold reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN


class CircuitBreakerConfig(TypedDict, total=False):
    """Allowed configuration when lazily creating a breaker."""

    failure_threshold: int
    recovery_timeout: float


class CircuitBreakerRegistry:
    """Global registry of circuit breakers per provider."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str, **kwargs: CircuitBreakerConfig) -> CircuitBreaker:
        """Get or create a circuit breaker for the named provider."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, **kwargs)
            return self._breakers[name]

    def reset(self, name: str) -> None:
        """Reset a specific circuit breaker."""
        with self._lock:
            if name in self._breakers:
                self._breakers[name].record_success()

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self._breakers.values():
                cb.record_success()


# Global singleton
_registry: CircuitBreakerRegistry | None = None


def get_registry() -> CircuitBreakerRegistry:
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry
