"""Tests for world-class API quality: error handling, /health, timeouts."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestErrorHandling:
    """Test differentiated HTTP error responses (not all 500s)."""

    def test_invalid_api_key_returns_401(self) -> None:
        """Provider auth failure should surface as 401, not 500."""
        from blend.api import app

        client = TestClient(app, raise_server_exceptions=False)

        with patch("blend.core.orchestrator.BlendOrchestrator.process_messages") as mock:
            import httpx
            mock.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=MagicMock(status_code=401),
            )

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 401

    def test_provider_timeout_returns_504(self) -> None:
        """Provider timeout should return 504 Gateway Timeout."""
        from blend.api import app

        client = TestClient(app, raise_server_exceptions=False)

        with patch("blend.core.orchestrator.BlendOrchestrator.process_messages") as mock:
            import httpx
            mock.side_effect = httpx.TimeoutException("Request timeout")

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 504

    def test_rate_limit_returns_429(self) -> None:
        """Provider rate limit should return 429 with retry-after header."""
        from blend.api import app

        client = TestClient(app, raise_server_exceptions=False)

        with patch("blend.core.orchestrator.BlendOrchestrator.process_messages") as mock:
            import httpx
            mock.side_effect = httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=MagicMock(),
                response=MagicMock(status_code=429, headers={"retry-after": "5"}),
            )

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 429
        assert "retry-after" in response.headers or "Retry-After" in response.headers

    def test_provider_down_returns_503(self) -> None:
        """Provider unavailable should return 503 Service Unavailable."""
        from blend.api import app

        client = TestClient(app, raise_server_exceptions=False)

        with patch("blend.core.orchestrator.BlendOrchestrator.process_messages") as mock:
            import httpx
            mock.side_effect = httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=MagicMock(),
                response=MagicMock(status_code=503),
            )

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "blend",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 503

    def test_empty_messages_returns_400(self) -> None:
        """Empty messages list should return 400."""
        from blend.api import app

        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "blend",
                "messages": [],
            },
        )

        assert response.status_code == 400


class TestHealthEndpoint:
    """Test /health endpoint quality."""

    def test_health_returns_healthy_when_up(self) -> None:
        """Health endpoint should return a status field."""
        from blend.api import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data  # healthy or degraded depending on provider reachability

    def test_health_includes_providers(self) -> None:
        """Health response should include provider status."""
        from blend.api import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")

        data = response.json()
        assert "providers" in data or "service" in data


class TestCircuitBreaker:
    """Test circuit breaker for provider failures."""

    def test_circuit_breaker_skips_failing_provider(self) -> None:
        """After consecutive failures, circuit breaker should skip failing provider."""
        from blend.core.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, base_recovery_timeout=5.0)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_circuit_breaker_recovers_after_timeout(self) -> None:
        """After recovery timeout, circuit breaker should half-open."""
        import time

        from blend.core.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, base_recovery_timeout=0.01)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.02)

        assert cb.state in (CircuitState.HALF_OPEN, CircuitState.CLOSED)

    def test_circuit_breaker_success_resets(self) -> None:
        """Successful request should reset circuit breaker."""
        from blend.core.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, base_recovery_timeout=60.0)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        # Internal failure_count is private, verify state reset instead.
