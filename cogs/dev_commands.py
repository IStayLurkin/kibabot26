import os
import sys
import json
import subprocess
import discord
from discord.ext import commands
from pathlib import Path

RESTART_STATE_FILE = Path(__file__).parent.parent / ".restart_state.json"


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

        msg = await ctx.send("```\n[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0%  Restarting...\n```")
        RESTART_STATE_FILE.write_text(json.dumps({
            "channel_id": ctx.channel.id,
            "message_id": msg.id,
        }))
        await self.bot.close()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    @commands.command(name="log")
    @commands.is_owner()
    async def show_log(self, ctx, *args):
        """Show bot.log. Usage: !log [lines] [errors] [filter]
        Examples: !log | !log 100 | !log errors | !log errors wan | !log errors 100 wan"""
        log_path = Path(__file__).parent.parent / "bot.log"
        if not log_path.exists():
            await ctx.send("No log file found. Make sure the bot was started via `start_bot.ps1`.")
            return

        lines = 50
        errors_only = False
        filter_term = None

        for arg in args:
            if arg.isdigit():
                lines = int(arg)
            elif arg.lower() == "errors":
                errors_only = True
            else:
                filter_term = arg.lower()

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        if errors_only:
            matched = [l for l in all_lines if " ERROR " in l or " WARNING " in l or "Traceback" in l or "Exception" in l]
        else:
            matched = all_lines

        if filter_term:
            matched = [l for l in matched if filter_term in l.lower()]

        tail_lines = matched[-lines:]
        tail = "".join(tail_lines)

        if not tail.strip():
            qualifier = "errors" if errors_only else "entries"
            suffix = f" matching '{filter_term}'" if filter_term else ""
            await ctx.send(f"No {qualifier}{suffix} found in the last {lines} lines.")
            return

        chunk_size = 1900
        chunks = [tail[i:i+chunk_size] for i in range(0, len(tail), chunk_size)]
        for chunk in chunks:
            await ctx.send(f"```\n{chunk}\n```")

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
    @show_log.error
    async def dev_command_error(self, ctx, error):
        if isinstance(error, commands.NotOwner):
            await ctx.send("You are not allowed to use that command.")
            return

        raise error


async def setup(bot):
    await bot.add_cog(DevCommands(bot))
