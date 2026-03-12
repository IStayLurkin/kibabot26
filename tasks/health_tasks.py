from discord.ext import tasks

from core.logging_config import get_logger
from services.performance_service import monitor_event_loop

logger = get_logger(__name__)


class HealthTasks:
    def __init__(self, bot):
        self.bot = bot
        self.loop_monitor_task = None

    def start_all(self):
        if not self.bot_health_check.is_running():
            self.bot_health_check.start()

        if self.loop_monitor_task is None or self.loop_monitor_task.done():
            self.loop_monitor_task = self.bot.loop.create_task(
                monitor_event_loop(self.bot.performance_tracker)
            )

    def stop_all(self):
        if self.bot_health_check.is_running():
            self.bot_health_check.cancel()

        if self.loop_monitor_task is not None and not self.loop_monitor_task.done():
            self.loop_monitor_task.cancel()

    def _format_brief_metric(self, label: str, current: float, average: float, maximum: float) -> str:
        return f"{label}={current:.0f}ms avg={average:.0f} max={maximum:.0f}"

    def _format_top_commands(self, commands_snapshot: list[dict]) -> str:
        if not commands_snapshot:
            return ""

        top_commands = ", ".join(
            f"{item['name']} {item['avg_ms']:.0f}ms"
            for item in commands_snapshot[:2]
        )
        return f"commands=[{top_commands}]"

    def _format_top_services(self, services_snapshot: list[dict]) -> str:
        if not services_snapshot:
            return ""

        top_services = ", ".join(
            f"{item['name']} {item['avg_ms']:.0f}ms"
            for item in services_snapshot[:2]
        )
        return f"services=[{top_services}]"

    def _format_slow_ops(self, slow_operations: list[dict]) -> str:
        if not slow_operations:
            return ""

        slow_summary = ", ".join(
            f"{item['category']}:{item['name']} {item['duration_ms']:.0f}ms"
            for item in slow_operations[:2]
        )
        return f"slow=[{slow_summary}]"

    @tasks.loop(minutes=1)
    async def bot_health_check(self):
        tracker = getattr(self.bot, "performance_tracker", None)
        websocket_latency_ms = round(self.bot.latency * 1000, 2)

        if tracker is not None:
            tracker.record_websocket_latency(websocket_latency_ms)
            snapshot = tracker.get_health_snapshot()
        else:
            snapshot = {
                "websocket_current_ms": websocket_latency_ms,
                "websocket_avg_ms": websocket_latency_ms,
                "websocket_max_ms": websocket_latency_ms,
                "loop_lag_avg_ms": 0.0,
                "loop_lag_max_ms": 0.0,
                "commands": [],
                "services": [],
                "runtime_services": [],
                "slow_operations": [],
            }

        parts = [
            f"guilds={len(self.bot.guilds)}",
            self._format_brief_metric(
                "ws",
                snapshot["websocket_current_ms"],
                snapshot["websocket_avg_ms"],
                snapshot["websocket_max_ms"],
            ),
        ]

        if snapshot["loop_lag_max_ms"] >= 10:
            parts.append(
                self._format_brief_metric(
                    "loop",
                    snapshot["loop_lag_avg_ms"],
                    snapshot["loop_lag_avg_ms"],
                    snapshot["loop_lag_max_ms"],
                )
            )

        command_summary = self._format_top_commands(snapshot["commands"])
        if command_summary:
            parts.append(command_summary)

        service_summary = self._format_top_services(snapshot["runtime_services"])
        if service_summary:
            parts.append(service_summary)

        slow_summary = self._format_slow_ops(snapshot["slow_operations"])
        if slow_summary:
            parts.append(slow_summary)

        logger.info("[health] %s", " | ".join(parts))

    @bot_health_check.before_loop
    async def before_bot_health_check(self):
        await self.bot.wait_until_ready()
