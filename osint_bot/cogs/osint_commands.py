from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from osint_bot.core.constants import SAFE_USAGE_POLICY
from osint_bot.services.formatting import build_discord_payload
from osint_bot.services.models import OSINTRequest


class OSINTCommands(commands.Cog):
    osint = app_commands.Group(name="osint", description="Safe OSINT assistant commands.")

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.osint_service = getattr(bot, "osint_service")

    async def _run_request(
        self,
        send_callable,
        *,
        requester_name: str,
        requester_id: int | None,
        target_type: str,
        target_value: str,
        authorization: bool = False,
        mode: str = "public_enrichment",
    ) -> None:
        try:
            result = await self.osint_service.handle_request(
                OSINTRequest(
                    target_type=target_type,
                    target_value=target_value,
                    mode=mode,
                    authorization=authorization,
                    requester_name=requester_name,
                    requester_id=requester_id,
                )
            )
            message, attachment = build_discord_payload(result)
            await send_callable(message, file=attachment)
        except ValueError as exc:
            await send_callable(f"Validation error: {exc}")
        except Exception as exc:
            await send_callable(f"OSINT lookup failed: {exc}")

    @commands.group(name="osint", invoke_without_command=True)
    async def osint_group(self, ctx: commands.Context) -> None:
        await ctx.send(
            "Usage: `!osint domain <domain>`, `!osint url <url>`, `!osint ip <ip>`, "
            "`!osint username <handle>`, `!osint summarize <text>`, or `!osint policy`"
        )

    @osint_group.command(name="domain")
    async def osint_domain(self, ctx: commands.Context, domain: str, authorized: bool = False) -> None:
        async with ctx.typing():
            await self._run_request(
                ctx.send,
                requester_name=ctx.author.display_name,
                requester_id=ctx.author.id,
                target_type="domain",
                target_value=domain,
                authorization=authorized,
                mode="owned_asset_check" if authorized else "public_enrichment",
            )

    @osint_group.command(name="url")
    async def osint_url(self, ctx: commands.Context, url: str, authorized: bool = False) -> None:
        async with ctx.typing():
            await self._run_request(
                ctx.send,
                requester_name=ctx.author.display_name,
                requester_id=ctx.author.id,
                target_type="url",
                target_value=url,
                authorization=authorized,
                mode="owned_asset_check" if authorized else "public_enrichment",
            )

    @osint_group.command(name="ip")
    async def osint_ip(self, ctx: commands.Context, ip_value: str, authorized: bool = False) -> None:
        async with ctx.typing():
            await self._run_request(
                ctx.send,
                requester_name=ctx.author.display_name,
                requester_id=ctx.author.id,
                target_type="ip",
                target_value=ip_value,
                authorization=authorized,
                mode="owned_asset_check" if authorized else "public_enrichment",
            )

    @osint_group.command(name="username")
    async def osint_username(self, ctx: commands.Context, username: str) -> None:
        async with ctx.typing():
            await self._run_request(
                ctx.send,
                requester_name=ctx.author.display_name,
                requester_id=ctx.author.id,
                target_type="username",
                target_value=username,
                mode="public_enrichment",
            )

    @osint_group.command(name="summarize")
    async def osint_summarize(self, ctx: commands.Context, *, text: str) -> None:
        async with ctx.typing():
            await self._run_request(
                ctx.send,
                requester_name=ctx.author.display_name,
                requester_id=ctx.author.id,
                target_type="text",
                target_value=text,
                mode="summarize_only",
            )

    @osint_group.command(name="policy")
    async def osint_policy(self, ctx: commands.Context) -> None:
        await ctx.send(SAFE_USAGE_POLICY)

    @osint.command(name="domain", description="Safe domain enrichment or owned-asset checks.")
    async def slash_domain(
        self,
        interaction: discord.Interaction,
        domain: str,
        authorized: bool = False,
    ) -> None:
        await interaction.response.defer(thinking=True)
        await self._run_request(
            interaction.followup.send,
            requester_name=interaction.user.display_name,
            requester_id=interaction.user.id,
            target_type="domain",
            target_value=domain,
            authorization=authorized,
            mode="owned_asset_check" if authorized else "public_enrichment",
        )

    @osint.command(name="url", description="Safe URL enrichment or owned-asset checks.")
    async def slash_url(
        self,
        interaction: discord.Interaction,
        url: str,
        authorized: bool = False,
    ) -> None:
        await interaction.response.defer(thinking=True)
        await self._run_request(
            interaction.followup.send,
            requester_name=interaction.user.display_name,
            requester_id=interaction.user.id,
            target_type="url",
            target_value=url,
            authorization=authorized,
            mode="owned_asset_check" if authorized else "public_enrichment",
        )

    @osint.command(name="ip", description="Safe IP enrichment or owned-asset checks.")
    async def slash_ip(
        self,
        interaction: discord.Interaction,
        ip: str,
        authorized: bool = False,
    ) -> None:
        await interaction.response.defer(thinking=True)
        await self._run_request(
            interaction.followup.send,
            requester_name=interaction.user.display_name,
            requester_id=interaction.user.id,
            target_type="ip",
            target_value=ip,
            authorization=authorized,
            mode="owned_asset_check" if authorized else "public_enrichment",
        )

    @osint.command(name="username", description="Public username enrichment guidance.")
    async def slash_username(self, interaction: discord.Interaction, username: str) -> None:
        await interaction.response.defer(thinking=True)
        await self._run_request(
            interaction.followup.send,
            requester_name=interaction.user.display_name,
            requester_id=interaction.user.id,
            target_type="username",
            target_value=username,
            mode="public_enrichment",
        )

    @osint.command(name="summarize", description="Summarize user-provided public text.")
    async def slash_summarize(self, interaction: discord.Interaction, text: str) -> None:
        await interaction.response.defer(thinking=True)
        await self._run_request(
            interaction.followup.send,
            requester_name=interaction.user.display_name,
            requester_id=interaction.user.id,
            target_type="text",
            target_value=text,
            mode="summarize_only",
        )

    @osint.command(name="policy", description="Show the OSINT bot safety policy.")
    async def slash_policy(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(SAFE_USAGE_POLICY, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    cog = OSINTCommands(bot)
    await bot.add_cog(cog)
    bot.tree.add_command(cog.osint)
