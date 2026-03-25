import asyncio

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
            self.loop_monitor_task = asyncio.get_running_loop().create_task(
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

        logger.info("Ping: %sms", f"{websocket_latency_ms:.0f}")

    @bot_health_check.before_loop
    async def before_bot_health_check(self):
        await self.bot.wait_until_ready()
