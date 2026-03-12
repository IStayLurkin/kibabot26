from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass

from core.logging_config import get_logger

logger = get_logger(__name__)

COMMAND_INFO_THRESHOLD_MS = 1000
COMMAND_WARNING_THRESHOLD_MS = 3000
COMMAND_ERROR_THRESHOLD_MS = 8000

SERVICE_INFO_THRESHOLD_MS = 3000
SERVICE_WARNING_THRESHOLD_MS = 5000
SERVICE_ERROR_THRESHOLD_MS = 8000


@dataclass(slots=True)
class OperationRecord:
    name: str
    duration_ms: float
    category: str
    severity: str
    created_at: float


class PerformanceTracker:
    def __init__(self) -> None:
        self.websocket_latency_samples_ms = deque(maxlen=120)
        self.command_durations_ms: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=50))
        self.service_durations_ms: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=50))
        self.recent_slow_operations = deque(maxlen=25)
        self.command_start_times: dict[int, tuple[str, float]] = {}
        self.loop_lag_samples_ms = deque(maxlen=120)
        self.last_health_snapshot_at = 0.0

    def record_websocket_latency(self, latency_ms: float) -> None:
        self.websocket_latency_samples_ms.append(float(latency_ms))

    def record_loop_lag(self, lag_ms: float) -> None:
        self.loop_lag_samples_ms.append(float(lag_ms))

    def start_command(self, command_id: int, command_name: str) -> None:
        self.command_start_times[command_id] = (command_name, time.perf_counter())

    def finish_command(self, command_id: int) -> float | None:
        started = self.command_start_times.pop(command_id, None)
        if started is None:
            return None

        command_name, started_at = started
        duration_ms = (time.perf_counter() - started_at) * 1000
        self.command_durations_ms[command_name].append(duration_ms)
        self._record_slow_operation("command", command_name, duration_ms)
        return duration_ms

    def record_service_call(self, name: str, duration_ms: float) -> None:
        self.service_durations_ms[name].append(float(duration_ms))
        self._record_slow_operation("service", name, duration_ms)

    @asynccontextmanager
    async def track_service_call(self, name: str):
        started_at = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000
            self.record_service_call(name, duration_ms)

    def _record_slow_operation(
        self,
        category: str,
        name: str,
        duration_ms: float,
    ) -> None:
        if name.startswith("startup."):
            return

        severity = self._get_severity(duration_ms, category)
        if severity is None:
            return

        record = OperationRecord(
            name=name,
            duration_ms=round(duration_ms, 2),
            category=category,
            severity=severity,
            created_at=time.time(),
        )
        self.recent_slow_operations.append(record)
        message = "[perf] status=success category=%s name=%s duration_ms=%.2f severity=%s"

        if severity == "critical_slow":
            logger.warning(message, category, name, duration_ms, severity)
            return

        logger.info(message, category, name, duration_ms, severity)

    def _get_severity(self, duration_ms: float, category: str = "service") -> str | None:
        if category == "command":
            info_threshold = COMMAND_INFO_THRESHOLD_MS
            warning_threshold = COMMAND_WARNING_THRESHOLD_MS
            error_threshold = COMMAND_ERROR_THRESHOLD_MS
        else:
            info_threshold = SERVICE_INFO_THRESHOLD_MS
            warning_threshold = SERVICE_WARNING_THRESHOLD_MS
            error_threshold = SERVICE_ERROR_THRESHOLD_MS

        if duration_ms >= error_threshold:
            return "critical_slow"

        if duration_ms >= warning_threshold:
            return "slow"

        if duration_ms >= info_threshold:
            return "elevated"

        return None

    def _average(self, values: deque[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _max(self, values: deque[float]) -> float:
        if not values:
            return 0.0
        return max(values)

    def get_health_snapshot(self) -> dict:
        websocket_avg_ms = self._average(self.websocket_latency_samples_ms)
        websocket_max_ms = self._max(self.websocket_latency_samples_ms)
        loop_lag_avg_ms = self._average(self.loop_lag_samples_ms)
        loop_lag_max_ms = self._max(self.loop_lag_samples_ms)

        recent_commands = []
        for name, samples in self.command_durations_ms.items():
            if not samples:
                continue
            recent_commands.append({
                "name": name,
                "avg_ms": round(self._average(samples), 2),
                "max_ms": round(self._max(samples), 2),
                "count": len(samples),
            })
        recent_commands.sort(key=lambda item: item["avg_ms"], reverse=True)

        recent_services = []
        for name, samples in self.service_durations_ms.items():
            if not samples:
                continue
            recent_services.append({
                "name": name,
                "avg_ms": round(self._average(samples), 2),
                "max_ms": round(self._max(samples), 2),
                "count": len(samples),
            })
        recent_services.sort(key=lambda item: item["avg_ms"], reverse=True)

        runtime_services = [
            item for item in recent_services
            if not item["name"].startswith("startup.")
        ]

        slow_ops = [
            {
                "category": record.category,
                "name": record.name,
                "duration_ms": record.duration_ms,
                "severity": record.severity,
            }
            for record in list(self.recent_slow_operations)[-5:]
        ]

        return {
            "websocket_current_ms": round(self.websocket_latency_samples_ms[-1], 2) if self.websocket_latency_samples_ms else 0.0,
            "websocket_avg_ms": round(websocket_avg_ms, 2),
            "websocket_max_ms": round(websocket_max_ms, 2),
            "loop_lag_avg_ms": round(loop_lag_avg_ms, 2),
            "loop_lag_max_ms": round(loop_lag_max_ms, 2),
            "commands": recent_commands[:5],
            "services": recent_services[:8],
            "runtime_services": runtime_services[:8],
            "slow_operations": slow_ops,
        }


async def monitor_event_loop(tracker: PerformanceTracker, *, interval_seconds: float = 5.0) -> None:
    loop = asyncio.get_running_loop()

    while True:
        target = loop.time() + interval_seconds
        await asyncio.sleep(interval_seconds)
        drift_seconds = loop.time() - target
        tracker.record_loop_lag(max(drift_seconds * 1000, 0.0))
