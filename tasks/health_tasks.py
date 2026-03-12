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
                "slow_operations": [],
            }

        command_summary = ", ".join(
            f"{item['name']} avg={item['avg_ms']} max={item['max_ms']}"
            for item in snapshot["commands"][:3]
        ) or "none"

        service_summary = ", ".join(
            f"{item['name']} avg={item['avg_ms']} max={item['max_ms']}"
            for item in snapshot["services"][:4]
        ) or "none"

        slow_summary = ", ".join(
            f"{item['category']}:{item['name']}={item['duration_ms']}ms"
            for item in snapshot["slow_operations"]
        ) or "none"

        logger.info(
            "[health] guilds=%s websocket_latency_ms=%.2f websocket_avg_ms=%.2f websocket_max_ms=%.2f "
            "loop_lag_avg_ms=%.2f loop_lag_max_ms=%.2f top_commands=%s top_services=%s slow_ops=%s",
            len(self.bot.guilds),
            snapshot["websocket_current_ms"],
            snapshot["websocket_avg_ms"],
            snapshot["websocket_max_ms"],
            snapshot["loop_lag_avg_ms"],
            snapshot["loop_lag_max_ms"],
            command_summary,
            service_summary,
            slow_summary,
        )

    @bot_health_check.before_loop
    async def before_bot_health_check(self):
        await self.bot.wait_until_ready()
