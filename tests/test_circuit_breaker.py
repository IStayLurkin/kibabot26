import pytest
import time
from services.circuit_breaker import CircuitBreaker

def test_circuit_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)
    assert cb.is_available() is True

def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_available() is False

def test_circuit_resets_on_success():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.is_available() is True

def test_circuit_recovers_after_cooldown():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_available() is False
    time.sleep(0.01)
    assert cb.is_available() is True
