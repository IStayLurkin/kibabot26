import os
import sys
import subprocess
import discord
from discord.ext import commands


class DevCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    def managed_extensions(self):
        return [
            "cogs.expense_commands",
            "cogs.budget_commands",
            "cogs.error_handler",
            "cogs.chat_commands",
            "cogs.dev_commands",
            "cogs.media_commands",
            "cogs.agent_commands",
            "cogs.runtime_commands",
            "cogs.code_commands",
            "cogs.video_commands",
            "tasks.vram_guard",
        ]

    @commands.command()
    @commands.is_owner()
    async def reload(self, ctx, extension: str):
        extension = extension.strip().replace(".py", "")
        full_extension = f"cogs.{extension}"

        try:
            await self.bot.reload_extension(full_extension)
            await ctx.send(f"Reloaded `{full_extension}`")
        except Exception as exc:
            await ctx.send(
                f"Failed to reload `{full_extension}`: `{type(exc).__name__}: {exc}`"
            )

    @commands.command()
    @commands.is_owner()
    async def reloadall(self, ctx):
        results = []

        for extension in self.managed_extensions():
            try:
                await self.bot.reload_extension(extension)
                results.append(f"Reloaded `{extension}`")
            except Exception as exc:
                results.append(
                    f"Failed `{extension}`: `{type(exc).__name__}: {exc}`"
                )

        await ctx.send("\n".join(results))

    @commands.command()
    @commands.is_owner()
    async def whichmodel(self, ctx):
        runtime_service = getattr(self.bot, "model_runtime_service", None)
        if runtime_service is None:
            await ctx.send("Runtime model service is not available.")
            return

        await ctx.send(runtime_service.get_current_model_text("llm"))

    @commands.command()
    @commands.is_owner()
    async def cogs(self, ctx):
        loaded = sorted(self.bot.extensions.keys())

        if not loaded:
            await ctx.send("No cogs are currently loaded.")
            return

        await ctx.send("Loaded cogs:\n" + "\n".join(f"`{name}`" for name in loaded))

    @commands.command()
    @commands.is_owner()
    async def update(self, ctx):
        """Pull latest code from git and restart the bot."""
        try:
            result = subprocess.run(
                ["git", "pull"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            output = (result.stdout + result.stderr).strip() or "No output."
        except Exception as exc:
            await ctx.send(f"Git pull failed: `{exc}`")
            return

        if result.returncode != 0:
            await ctx.send(f"Git pull failed (exit {result.returncode}):\n```{output}```")
            return

        await ctx.send(f"```{output}```\nRestarting...")
        await self.bot.close()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    @commands.command()
    @commands.is_owner()
    async def reloadchat(self, ctx):
        try:
            await self.bot.reload_extension("cogs.chat_commands")
            await ctx.send("Reloaded `cogs.chat_commands`")
        except Exception as exc:
            await ctx.send(f"Failed: `{type(exc).__name__}: {exc}`")

    @reload.error
    @reloadall.error
    @whichmodel.error
    @cogs.error
    @update.error
    @reloadchat.error
    async def dev_command_error(self, ctx, error):
        if isinstance(error, commands.NotOwner):
            await ctx.send("You are not allowed to use that command.")
            return

        raise error


async def setup(bot):
    await bot.add_cog(DevCommands(bot))
