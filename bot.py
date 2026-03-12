import asyncio
import discord
from discord.ext import commands
from database.database import init_db
from core.config import (
    DISCORD_BOT_TOKEN,
    BOT_PREFIX,
    BOT_TIMEZONE,
)
from core.logging_config import setup_logging, get_logger
from tasks.task_manager import TaskManager
from services.llm_service import LLMService

setup_logging()
logger = get_logger(__name__)

intents = discord.Intents.default()
intents.message_content = True


class ExpenseBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_banner_printed = False
        self.task_manager = TaskManager(self)
        self.llm_service = None

    async def setup_hook(self):
        logger.info("setup_hook: initializing database...")
        await init_db()

        logger.info("setup_hook: initializing llm service...")
        self.llm_service = LLMService()

        logger.info("setup_hook: loading expense commands cog...")
        await self.load_extension("cogs.expense_commands")

        logger.info("setup_hook: loading error handler cog...")
        await self.load_extension("cogs.error_handler")

        logger.info("setup_hook: loading chat commands cog...")
        await self.load_extension("cogs.chat_commands")

        logger.info("setup_hook: loading dev commands cog...")
        await self.load_extension("cogs.dev_commands")

        logger.info("setup_hook: loading media commands cog...")
        await self.load_extension("cogs.media_commands")

        logger.info("setup_hook: loading agent commands cog...")
        await self.load_extension("cogs.agent_commands")

        logger.info("setup_hook: starting background tasks...")
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

        logger.info("=" * 52)
        logger.info("Kiba Bot Startup")
        logger.info("=" * 52)
        logger.info("User: %s", self.user)
        logger.info("Prefix: %s", BOT_PREFIX)
        logger.info("Provider: %s", provider)
        logger.info("Model: %s", model)
        logger.info("Timezone: %s", BOT_TIMEZONE)
        logger.info("Database: connected")
        logger.info("Cogs loaded: %s", len(loaded_cogs))

        for cog in loaded_cogs:
            logger.info("  - %s", cog)

        logger.info("=" * 52)


bot = ExpenseBot(command_prefix=BOT_PREFIX, intents=intents, help_command=None)


@bot.event
async def on_ready():
    logger.info("Bot connected as %s", bot.user)

    if not bot.startup_banner_printed:
        bot.print_startup_banner()
        bot.startup_banner_printed = True
    else:
        logger.info("on_ready fired again after reconnect/resume.")


@bot.event
async def on_resumed():
    logger.info("Discord session resumed successfully.")


@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord gateway.")


async def main():
    if not DISCORD_BOT_TOKEN:
        raise ValueError("Error: DISCORD_BOT_TOKEN not found in environment variables.")

    logger.info("Starting bot...")

    async with bot:
        await bot.start(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())