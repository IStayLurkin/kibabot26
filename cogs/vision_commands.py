from __future__ import annotations

import re

from discord.ext import commands

from core.config import VISION_FAST_MODEL, VISION_BEST_MODEL
from core.logging_config import get_logger

logger = get_logger(__name__)


class VisionCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.group(name="vision", invoke_without_command=True, help="Analyze an image with a vision model.")
    async def vision_group(self, ctx: commands.Context, *, prompt: str = ""):
        await ctx.send(
            f"Usage: `!vision fast [prompt]` or `!vision best [prompt]` — attach an image or include a URL.\n"
            f"fast → `{VISION_FAST_MODEL}` | best → `{VISION_BEST_MODEL}`"
        )

    @vision_group.command(name="fast", help="Analyze with fast vision model.")
    async def vision_fast(self, ctx: commands.Context, *, prompt: str = ""):
        await self._run_vision(ctx, prompt, "fast")

    @vision_group.command(name="best", help="Analyze with best vision model.")
    async def vision_best(self, ctx: commands.Context, *, prompt: str = ""):
        await self._run_vision(ctx, prompt, "best")

    async def _run_vision(self, ctx: commands.Context, prompt: str, tier: str):
        service = getattr(self.bot, "vision_service", None)
        if service is None:
            await ctx.send("Vision service is not available.")
            return

        # Get image from attachment or URL in prompt
        image_bytes = None
        image_url = None

        if ctx.message.attachments:
            att = ctx.message.attachments[0]
            if att.content_type and att.content_type.startswith("image/"):
                image_bytes = await att.read()
                content_type = att.content_type
            else:
                content_type = "image/png"
        else:
            content_type = "image/png"

        if image_bytes is None:
            # Try to find URL in prompt
            url_match = re.search(r"https?://\S+\.(?:png|jpg|jpeg|gif|webp)", prompt, re.IGNORECASE)
            if url_match:
                image_url = url_match.group(0)
                prompt = prompt.replace(image_url, "").strip()

        if image_bytes is None and image_url is None:
            await ctx.send("Attach an image or include an image URL.")
            return

        async with ctx.typing():
            try:
                if image_bytes is not None:
                    result = await service.analyze_bytes(
                        image_bytes,
                        prompt=prompt,
                        content_type=content_type,
                        tier=tier,
                    )
                else:
                    result = await service.analyze_url(image_url, prompt=prompt, tier=tier)

                if not result:
                    await ctx.send("Vision model returned no response.")
                    return
                if len(result) <= 1900:
                    await ctx.send(result)
                else:
                    for i in range(0, len(result), 1900):
                        await ctx.send(result[i:i + 1900])
            except Exception as exc:
                logger.error("[vision] Error: %s", exc)
                await ctx.send(f"Vision error: {exc}")


async def setup(bot: commands.Bot):
    await bot.add_cog(VisionCommands(bot))
