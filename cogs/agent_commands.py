from __future__ import annotations

import logging

from discord.ext import commands

from core.feature_flags import AGENT_ENABLED, OSINT_ENABLED
from services.agent_service import AgentService
from services.osint_service import OSINTService

logger = logging.getLogger(__name__)


def is_owner_or_admin():
    async def predicate(ctx: commands.Context) -> bool:
        if await ctx.bot.is_owner(ctx.author):
            return True

        perms = getattr(ctx.author, "guild_permissions", None)
        return bool(perms and perms.administrator)

    return commands.check(predicate)


class AgentCommands(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        llm_service = getattr(bot, "llm_service", None)
        self.agent_service = AgentService(llm_service=llm_service)
        self.osint_service = getattr(
            bot,
            "osint_service",
            OSINTService(performance_tracker=getattr(bot, "performance_tracker", None)),
        )

    @commands.command(name="agent")
    @is_owner_or_admin()
    async def agent_command(self, ctx: commands.Context, action: str | None = None) -> None:
        if not AGENT_ENABLED:
            await ctx.send("Agent features are disabled.")
            return

        if ctx.guild is None:
            await ctx.send("This command must be used in a server channel.")
            return

        action = (action or "").strip().lower()
        if action not in {"on", "off", "status"}:
            await ctx.send("Usage: `!agent on`, `!agent off`, or `!agent status`")
            return

        guild_id = ctx.guild.id
        channel_id = ctx.channel.id

        if action == "on":
            self.agent_service.enable_channel(guild_id, channel_id)
            await ctx.send("Agent enabled in this channel.")
            return

        if action == "off":
            self.agent_service.disable_channel(guild_id, channel_id)
            await ctx.send("Agent disabled in this channel.")
            return

        status = self.agent_service.get_status(guild_id, channel_id)
        await ctx.send(f"Agent is currently **{status}** in this channel.")

    @commands.command(name="osint")
    async def osint_command(self, ctx: commands.Context, *, query: str) -> None:
        if not OSINT_ENABLED:
            await ctx.send("OSINT features are disabled.")
            return

        async with ctx.typing():
            try:
                result = await self.osint_service.lookup_query(query)
                await ctx.send(result)
            except Exception as exc:
                logger.exception("!osint failed")
                await ctx.send(f"OSINT lookup failed: {exc}")

    @commands.command(name="whois")
    async def whois_command(self, ctx: commands.Context, *, domain: str) -> None:
        if not OSINT_ENABLED:
            await ctx.send("OSINT features are disabled.")
            return

        async with ctx.typing():
            try:
                result = await self.osint_service.whois_lookup(domain)
                await ctx.send(f"```text\n{result[:3900]}\n```")
            except Exception as exc:
                logger.exception("!whois failed")
                await ctx.send(f"Whois lookup failed: {exc}")

    @commands.command(name="domain")
    async def domain_command(self, ctx: commands.Context, *, domain: str) -> None:
        if not OSINT_ENABLED:
            await ctx.send("OSINT features are disabled.")
            return

        async with ctx.typing():
            try:
                dns_result = await self.osint_service.dns_lookup(domain)
                ssl_result = await self.osint_service.ssl_lookup(domain)
                await ctx.send(f"```text\n{dns_result}\n\n{ssl_result}\n```")
            except Exception as exc:
                logger.exception("!domain failed")
                await ctx.send(f"Domain lookup failed: {exc}")

    @commands.Cog.listener()
    async def on_message(self, message) -> None:
        if not AGENT_ENABLED:
            return

        if message.author.bot:
            return

        try:
            await self.agent_service.maybe_handle_game_message(message)
        except Exception:
            logger.exception("Agent on_message handler failed")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AgentCommands(bot))
