from __future__ import annotations

from discord.ext import commands

from core.config import THINKING_FAST_MODEL, THINKING_BEST_MODEL
from core.logging_config import get_logger
from services.thinking_service import THINKING_TIERS

logger = get_logger(__name__)

MAX_THINK_LENGTH = 3800


class ThinkingCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.group(name="think", invoke_without_command=True, help="Run a reasoning/thinking model on a prompt.")
    async def think_group(self, ctx: commands.Context, *, prompt: str = ""):
        if not prompt:
            await ctx.send(
                f"Usage: `!think fast <prompt>` or `!think best <prompt>`\n"
                f"fast → `{THINKING_FAST_MODEL}` | best → `{THINKING_BEST_MODEL}`"
            )
            return
        # Default to fast tier if no subcommand
        await self._run_think(ctx, prompt, "fast")

    @think_group.command(name="fast", help=f"Think with {THINKING_FAST_MODEL} (faster).")
    async def think_fast(self, ctx: commands.Context, *, prompt: str):
        await self._run_think(ctx, prompt, "fast")

    @think_group.command(name="best", help=f"Think with {THINKING_BEST_MODEL} (deeper reasoning).")
    async def think_best(self, ctx: commands.Context, *, prompt: str):
        await self._run_think(ctx, prompt, "best")

    async def _run_think(self, ctx: commands.Context, prompt: str, tier: str):
        service = getattr(self.bot, "thinking_service", None)
        if service is None:
            await ctx.send("Thinking service is not available.")
            return
        if len(prompt) > MAX_THINK_LENGTH:
            await ctx.send(f"Prompt too long. Keep it under {MAX_THINK_LENGTH} characters.")
            return
        async with ctx.typing():
            try:
                result = await service.think(prompt, tier=tier)
                if not result:
                    await ctx.send("No response from thinking model.")
                    return
                # Send in chunks if long
                limit = 1900
                if len(result) <= limit:
                    await ctx.send(result)
                else:
                    for i in range(0, len(result), limit):
                        await ctx.send(result[i:i+limit])
            except Exception as exc:
                logger.error("[think] Error: %s", exc)
                await ctx.send(f"Thinking model error: {exc}")


async def setup(bot: commands.Bot):
    await bot.add_cog(ThinkingCommands(bot))
