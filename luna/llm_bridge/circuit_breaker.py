"""Circuit Breaker — protects Luna from cascading LLM failures.

Pattern: CLOSED -> OPEN (after 3 failures) -> HALF_OPEN (after timeout).

When OPEN, Luna freezes Psi (no drift from failed calls) and returns
gracefully degraded responses. After recovery_timeout, transitions to
HALF_OPEN to allow a single probe request.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

log = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Protects LLM calls with circuit breaker pattern.

    - CLOSED: Normal operation. Failures are counted.
    - OPEN: All requests rejected. Psi frozen.
    - HALF_OPEN: One probe request allowed. Success -> CLOSED, failure -> OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 300.0,  # 5 minutes
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._total_trips: int = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic OPEN -> HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                log.info("Circuit breaker: OPEN -> HALF_OPEN (%.0fs elapsed)", elapsed)
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        current = self.state  # triggers auto-transition
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            return True  # allow probe
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            log.info("Circuit breaker: HALF_OPEN -> CLOSED (probe succeeded)")
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            log.warning("Circuit breaker: HALF_OPEN -> OPEN (probe failed)")
            self._state = CircuitState.OPEN
            self._total_trips += 1
        elif self._failure_count >= self._failure_threshold:
            log.warning(
                "Circuit breaker: CLOSED -> OPEN (%d failures)",
                self._failure_count,
            )
            self._state = CircuitState.OPEN
            self._total_trips += 1

    def to_dict(self) -> dict:
        """Serialize for dashboard/API."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
            "total_trips": self._total_trips,
            "last_failure_time": self._last_failure_time,
        }
