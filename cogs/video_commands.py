from __future__ import annotations

import asyncio
import discord
from discord.ext import commands
from pathlib import Path

from core.config import MAX_PROMPT_LENGTH
from core.logging_config import get_logger

logger = get_logger(__name__)


class VideoCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    def _get_service(self, name: str):
        return getattr(self.bot, name, None)

    async def _handle_video(
        self,
        ctx: commands.Context,
        prompt: str,
        label: str,
        service_attr: str,
        generate_kwargs: dict,
    ):
        if not prompt.strip():
            await ctx.send(f"Provide a prompt. Example: `!{ctx.invoked_with} a sunset over the ocean`")
            return

        if len(prompt) > MAX_PROMPT_LENGTH:
            await ctx.send(f"Prompt too long. Keep it under {MAX_PROMPT_LENGTH} characters.")
            return

        async with self.bot.generating_lock:
            self.bot.generating_count += 1
            status_msg = await ctx.send(
                f"🎬 **Starting {label}...**\n[░░░░░░░░░░] 0%\n📟 **VRAM:** --"
            )

            _loop = asyncio.get_running_loop()

            def update_progress(percent: int, vram_gb: float):
                blocks = int(percent / 10)
                bar = "█" * blocks + "░" * (10 - blocks)
                asyncio.run_coroutine_threadsafe(
                    status_msg.edit(content=(
                        f"🎬 **Generating ({label})...**\n"
                        f"[{bar}] {percent}%\n"
                        f"📟 **VRAM:** {vram_gb}GB / 24.0GB"
                    )),
                    _loop,
                )

            try:
                service = self._get_service(service_attr)
                if service is None:
                    await status_msg.edit(content=f"❌ **{label} service not initialized.**")
                    return

                await status_msg.edit(content=f"🎬 **{label}** — killing Ollama, freeing RAM...")
                video_path = await service.generate(**generate_kwargs, callback=update_progress)

                if video_path and Path(video_path).exists():
                    await status_msg.edit(content=f"✅ **{label} complete!** Sending...")
                    await ctx.send(
                        content=f"🎬 **{label}** | Prompt: *{prompt[:100]}*",
                        file=discord.File(video_path, filename=Path(video_path).name),
                    )
                    await status_msg.edit(content=f"✅ **{label} complete!**")
                else:
                    await status_msg.edit(content=f"❌ **{label} failed.** Check VRAM and logs.")
            except Exception as exc:
                logger.exception("[video_commands] %s failed: %s", label, exc)
                exc_str = str(exc)[:1800]
                await status_msg.edit(content=f"❌ **{label} error:** {exc_str}")
            finally:
                self.bot.generating_count -= 1

    @commands.command(name="cogvideo2b")
    @commands.cooldown(1, 300, commands.BucketType.user)
    async def cogvideo2b(self, ctx: commands.Context, *, prompt: str = ""):
        """Generate a video with CogVideoX-2b (~12GB VRAM, ~3 min)."""
        await self._handle_video(ctx, prompt, "CogVideoX-2b", "cogvideo_service", {"model_size": "2b", "prompt": prompt})

    @commands.command(name="cogvideo5b")
    @commands.cooldown(1, 300, commands.BucketType.user)
    async def cogvideo5b(self, ctx: commands.Context, *, prompt: str = ""):
        """Generate a video with CogVideoX-5b (~24GB VRAM, ~8 min). Unloads Ollama first."""
        await self._handle_video(ctx, prompt, "CogVideoX-5b", "cogvideo_service", {"model_size": "5b", "prompt": prompt})

    @commands.command(name="animatediff")
    @commands.cooldown(1, 300, commands.BucketType.user)
    async def animatediff(self, ctx: commands.Context, *, prompt: str = ""):
        """Generate a video with AnimateDiff (~8GB VRAM, ~1 min)."""
        await self._handle_video(ctx, prompt, "AnimateDiff", "animatediff_service", {"prompt": prompt})

    @commands.command(name="wan")
    @commands.cooldown(1, 120, commands.BucketType.user)
    async def wan(self, ctx: commands.Context, *, prompt: str = ""):
        """Generate a 480p video with Wan2.1-1.3B (~6GB VRAM, ~2 min). Fast, fits in VRAM."""
        await self._handle_video(ctx, prompt, "Wan2.1-1.3B", "wan_fast_service", {"prompt": prompt})

    @commands.command(name="wan2")
    async def wan2(self, ctx: commands.Context, *, prompt: str = ""):
        """Wan2.2-TI2V-5B — disabled until flash_attn supports torch 2.10."""
        await ctx.send("❌ `!wan2` is temporarily disabled — Wan2.2 requires flash_attn which doesn't support torch 2.10 yet.")

    @commands.command(name="wan14b")
    @commands.cooldown(1, 600, commands.BucketType.user)
    async def wan14b(self, ctx: commands.Context, *, prompt: str = ""):
        """Generate a 720p video with Wan2.1-14B (~20GB VRAM, ~10 min). Unloads Ollama first."""
        await self._handle_video(ctx, prompt, "Wan2.1-14B", "wan_service", {"prompt": prompt})

    @cogvideo2b.error
    @cogvideo5b.error
    @animatediff.error
    @wan.error
    @wan2.error
    @wan14b.error
    async def video_cooldown_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"⏳ Cooldown — try again in {error.retry_after:.0f}s.")


async def setup(bot):
    await bot.add_cog(VideoCommands(bot))
