"""Tests for Circuit Breaker concurrency issues - Issue: Dynamic lockout state accuracy."""

import pytest
import threading
import time
from unittest.mock import patch

from blend.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    get_registry,
)


class TestCircuitBreakerConcurrency:
    """Test circuit breaker behavior under concurrent access."""

    def test_concurrent_record_failure_race_condition(self):
        """Concurrent failures may cause incorrect lockout_duration calculation."""
        cb = CircuitBreaker(name="test", failure_threshold=3, base_recovery_timeout=1.0)

        errors = []
        barrier = threading.Barrier(10)

        def record_fail():
            try:
                barrier.wait()
                for _ in range(5):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_fail) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Race conditions detected: {errors}"

        # The lockout_duration after concurrent failures may be incorrect
        # because multiple threads read-modify-write _failure_count and _lockout_duration
        print(f"Circuit state: {cb.state}, lockout: {cb._lockout_duration}")

    def test_concurrent_state_check_inaccuracy(self):
        """Circuit state check under concurrent access may return stale state."""
        cb = CircuitBreaker(name="test", failure_threshold=2, base_recovery_timeout=0.1)

        results = []
        barrier = threading.Barrier(5)

        def check_state():
            barrier.wait()
            for _ in range(10):
                state = cb.state  # Read state
                results.append(state)
                time.sleep(0.001)

        threads = [threading.Thread(target=check_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Under race conditions, some threads may see stale state
        # This is expected behavior but worth documenting
        print(f"State observations: {len(results)}, OPEN count: {results.count(CircuitState.OPEN)}")

    def test_registry_persistence_race(self):
        """Registry save_state may miss concurrent updates."""
        registry = get_registry()
        cb = registry.get("race_test", base_recovery_timeout=0.1)

        barrier = threading.Barrier(5)

        def trigger_and_save():
            barrier.wait()
            for _ in range(10):
                cb.record_failure()
            registry.save_state()

        threads = [threading.Thread(target=trigger_and_save) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check if saved state matches final state
        cb_after = registry.get("race_test")
        # The persisted state may not reflect the true final state
        print(f"Final lockout: {cb._lockout_duration}, Saved: {cb_after._lockout_duration}")

    def test_time_monotonic_vs_time_time(self):
        """Using time.monotonic() is correct, but state persistence breaks monotonicity."""
        cb = CircuitBreaker(name="test", base_recovery_timeout=1.0)

        # Record a failure
        cb.record_failure(error_code=401)  # Should set lockout to 3600

        # State is OPEN with lockout_duration = 3600
        assert cb._state == CircuitState.OPEN
        assert cb._lockout_duration == 3600

        # After restart (reload from persistence), time.monotonic() continues
        # but _last_failure_time was saved as time.monotonic() value
        # This should still work correctly if time.monotonic() is used consistently

        # Problem: if system time changes (not monotonic), recovery may be incorrect
        # But time.monotonic() is specifically designed to handle this
        print(f"Lockout duration correctly set to 3600s: {cb._lockout_duration == 3600}")


class TestCircuitBreakerDynamicLockout:
    """Test exponential backoff lockout behavior."""

    def test_exponential_backoff_calculation(self):
        """Test that exponential backoff is calculated correctly."""
        cb = CircuitBreaker(name="test", failure_threshold=1, base_recovery_timeout=1.0)

        # Trip the circuit
        cb.record_failure()

        # Should be OPEN
        assert cb._state == CircuitState.OPEN

        # First trip: base_recovery_timeout * 2^1 = 2.0s
        # (consecutive_trips starts at 0, increments to 1 on first trip)
        assert cb._lockout_duration == 2.0

    def test_consecutive_trips_exponential_growth(self):
        """Test that consecutive trips increase lockout exponentially."""
        cb = CircuitBreaker(name="test", failure_threshold=1, base_recovery_timeout=1.0)

        # Multiple consecutive trips
        for i in range(5):
            cb.record_failure()
            print(f"Trip {i+1}: lockout_duration = {cb._lockout_duration}s")

        # Should grow exponentially: 1 * 2^5 = 32s (capped at 86400)
        # But consecutive_trips may be wrong due to race condition
        print(f"Final lockout: {cb._lockout_duration}, consecutive_trips: {cb._consecutive_trips}")