from __future__ import annotations

import asyncio

import discord
from discord.ext import commands

from osint_bot.core.config import OSINT_BOT_PREFIX, OSINT_DISCORD_BOT_TOKEN
from osint_bot.core.logging_config import get_logger, setup_logging
from osint_bot.services.llm_service import OSINTLLMService
from osint_bot.services.osint_service import OSINTService

setup_logging()
logger = get_logger(__name__)


class OSINTBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=OSINT_BOT_PREFIX, intents=intents, help_command=None)
        self.osint_service = OSINTService(llm_service=OSINTLLMService())
        self.startup_banner_printed = False

    async def setup_hook(self) -> None:
        logger.info("Loading isolated OSINT commands...")
        await self.load_extension("osint_bot.cogs.osint_commands")

        try:
            synced = await self.tree.sync()
            logger.info("Synced %s application commands.", len(synced))
        except Exception:
            logger.exception("Failed to sync application commands.")

    def print_startup_banner(self) -> None:
        service = getattr(self, "osint_service", None)
        llm_service = getattr(service, "llm_service", None)
        provider = getattr(llm_service, "provider", "unknown")
        model = getattr(llm_service, "get_active_model_name", lambda: "unknown")()

        logger.info("=" * 52)
        logger.info("OSINT Bot Startup")
        logger.info("=" * 52)
        logger.info("User: %s", self.user)
        logger.info("Prefix: %sosint", OSINT_BOT_PREFIX)
        logger.info("Provider: %s", provider)
        logger.info("Model: %s", model)
        logger.info("Cogs loaded: %s", len(self.extensions))
        for cog in sorted(self.extensions.keys()):
            logger.info("  - %s", cog)
        logger.info("=" * 52)


bot = OSINTBot()


@bot.event
async def on_ready() -> None:
    logger.info("OSINT bot connected as %s", bot.user)
    if not bot.startup_banner_printed:
        bot.print_startup_banner()
        bot.startup_banner_printed = True


async def main() -> None:
    if not OSINT_DISCORD_BOT_TOKEN:
        raise ValueError("OSINT_DISCORD_BOT_TOKEN is not set.")

    async with bot:
        await bot.start(OSINT_DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())

