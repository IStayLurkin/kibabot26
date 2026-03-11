from discord.ext import tasks
from core.logging_config import get_logger

logger = get_logger(__name__)


class HealthTasks:
    def __init__(self, bot):
        self.bot = bot

    def start_all(self):
        if not self.bot_health_check.is_running():
            self.bot_health_check.start()

    def stop_all(self):
        if self.bot_health_check.is_running():
            self.bot_health_check.cancel()

    @tasks.loop(minutes=5)
    async def bot_health_check(self):
        logger.info(
            "[health] guilds=%s latency_ms=%s",
            len(self.bot.guilds),
            round(self.bot.latency * 1000)
        )

    @bot_health_check.before_loop
    async def before_bot_health_check(self):
        await self.bot.wait_until_ready()
