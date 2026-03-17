import time


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 120.0):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._opened_at: float | None = None

    def is_available(self) -> bool:
        if self._opened_at is None:
            return True
        elapsed = time.monotonic() - self._opened_at
        if elapsed > self._cooldown_seconds:
            self._failure_count = 0
            self._opened_at = None
            return True
        return False

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._opened_at = time.monotonic()

    def record_success(self):
        self._failure_count = 0
        self._opened_at = None
