import asyncio
import time

import discord
from discord.ext import commands

from core.config import (
    DISCORD_BOT_TOKEN,
    BOT_PREFIX,
    BOT_TIMEZONE,
)
from core.logging_config import setup_logging, get_logger
from database.database import init_db
from services.codegen_service import CodegenService
from services.image_service import ImageService
from services.llm_service import LLMService
from services.osint_service import OSINTService
from services.performance_service import PerformanceTracker
from services.video_service import VideoService
from services.voice_service import VoiceService
from tasks.task_manager import TaskManager

setup_logging()
logger = get_logger(__name__)

intents = discord.Intents.default()
intents.message_content = True


class ExpenseBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_banner_printed = False
        self.performance_tracker = PerformanceTracker()
        self.task_manager = TaskManager(self)
        self.llm_service = None
        self.image_service = None
        self.voice_service = None
        self.video_service = None
        self.codegen_service = None
        self.osint_service = None

    async def setup_hook(self):
        async with self.performance_tracker.track_service_call("startup.init_db"):
            await init_db()

        service_started_at = time.perf_counter()
        self.llm_service = LLMService(performance_tracker=self.performance_tracker)
        self.image_service = ImageService(
            llm_service=self.llm_service,
            performance_tracker=self.performance_tracker,
        )
        self.voice_service = VoiceService(
            llm_service=self.llm_service,
            performance_tracker=self.performance_tracker,
        )
        self.video_service = VideoService(
            llm_service=self.llm_service,
            performance_tracker=self.performance_tracker,
        )
        self.codegen_service = CodegenService(
            llm_service=self.llm_service,
            performance_tracker=self.performance_tracker,
        )
        self.osint_service = OSINTService(performance_tracker=self.performance_tracker)
        self.performance_tracker.record_service_call(
            "startup.init_services",
            (time.perf_counter() - service_started_at) * 1000,
        )

        extensions = [
            "cogs.expense_commands",
            "cogs.budget_commands",
            "cogs.error_handler",
            "cogs.chat_commands",
            "cogs.dev_commands",
            "cogs.media_commands",
            "cogs.agent_commands",
        ]

        for extension in extensions:
            logger.debug("Loading extension %s", extension)
            started_at = time.perf_counter()
            await self.load_extension(extension)
            self.performance_tracker.record_service_call(
                f"startup.load_extension.{extension}",
                (time.perf_counter() - started_at) * 1000,
            )

        self.task_manager.start_all()

    async def close(self):
        logger.info("Shutting down background tasks...")
        self.task_manager.stop_all()
        await super().close()

    def print_startup_banner(self):
        chat_cog = self.get_cog("ChatCommands")
        provider = "unknown"
        model = "unknown"

        if chat_cog is not None and getattr(chat_cog, "llm", None) is not None:
            provider = chat_cog.llm.provider
            model = chat_cog.llm._get_active_model_name()
        elif self.llm_service is not None:
            provider = getattr(self.llm_service, "provider", "unknown")
            get_model_name = getattr(self.llm_service, "_get_active_model_name", None)
            if callable(get_model_name):
                model = get_model_name()

        loaded_cogs = sorted(self.extensions.keys())
        startup_snapshot = self.performance_tracker.get_health_snapshot()
        logger.info(
            "[startup] user=%s prefix=%s provider=%s model=%s timezone=%s cogs=%s",
            self.user,
            BOT_PREFIX,
            provider,
            model,
            BOT_TIMEZONE,
            len(loaded_cogs),
        )
        logger.info(
            "[startup] ws=%sms services=%s commands=%s",
            f"{startup_snapshot['websocket_current_ms']:.0f}",
            len(startup_snapshot["services"]),
            len(startup_snapshot["commands"]),
        )
        logger.debug("Loaded cogs: %s", ", ".join(loaded_cogs))


bot = ExpenseBot(command_prefix=BOT_PREFIX, intents=intents, help_command=None)


@bot.event
async def on_ready():
    if not bot.startup_banner_printed:
        bot.print_startup_banner()
        bot.startup_banner_printed = True
    else:
        logger.info("[gateway] reconnected user=%s", bot.user)


@bot.event
async def on_resumed():
    logger.info("[gateway] session resumed")


@bot.event
async def on_disconnect():
    logger.warning("[gateway] disconnected")


@bot.before_invoke
async def before_any_command(ctx: commands.Context):
    if ctx.command is None:
        return

    bot.performance_tracker.start_command(id(ctx.message), ctx.command.qualified_name)


@bot.after_invoke
async def after_any_command(ctx: commands.Context):
    if ctx.command is None:
        return

    duration_ms = bot.performance_tracker.finish_command(id(ctx.message))
    if duration_ms is None:
        return

    if duration_ms >= 1000:
        logger.info(
            "[command] name=%s duration_ms=%.2f websocket_latency_ms=%.2f",
            ctx.command.qualified_name,
            duration_ms,
            bot.latency * 1000,
        )
    else:
        logger.debug(
            "[command] name=%s duration_ms=%.2f websocket_latency_ms=%.2f",
            ctx.command.qualified_name,
            duration_ms,
            bot.latency * 1000,
        )


async def main():
    if not DISCORD_BOT_TOKEN:
        raise ValueError("Error: DISCORD_BOT_TOKEN not found in environment variables.")

    logger.info("[startup] booting")

    async with bot:
        await bot.start(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
