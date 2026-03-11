import discord
from discord.ext import commands
from core.logging_config import get_logger

logger = get_logger(__name__)


class ErrorHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    async def send_error_embed(self, ctx, title: str, description: str):
        embed = discord.Embed(
            title=title,
            description=description,
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        if hasattr(ctx.command, "on_error"):
            return

        error = getattr(error, "original", error)

        if isinstance(error, commands.CommandNotFound):
            return

        if isinstance(error, commands.MissingRequiredArgument):
            command_name = ctx.command.qualified_name if ctx.command else "command"
            await self.send_error_embed(
                ctx,
                "Missing Argument",
                f"Missing required argument for `!{command_name}`."
            )
            return

        if isinstance(error, commands.BadArgument):
            command_name = ctx.command.qualified_name if ctx.command else "command"
            await self.send_error_embed(
                ctx,
                "Invalid Argument",
                f"Invalid argument provided for `!{command_name}`."
            )
            return

        if isinstance(error, commands.NotOwner):
            await self.send_error_embed(
                ctx,
                "Access Denied",
                "You are not allowed to use that command."
            )
            return

        if isinstance(error, commands.MissingPermissions):
            await self.send_error_embed(
                ctx,
                "Missing Permissions",
                "You do not have permission to use that command."
            )
            return

        logger.exception("Unhandled command error: %s", error)

        await self.send_error_embed(
            ctx,
            "Unexpected Error",
            "Something went wrong while running that command."
        )


async def setup(bot):
    await bot.add_cog(ErrorHandler(bot))
