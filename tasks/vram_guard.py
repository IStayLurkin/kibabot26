import torch
import gc
import asyncio
from discord.ext import tasks, commands
from core.logging_config import get_logger

logger = get_logger(__name__)

class VRAMGuard(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.hardware_service = getattr(bot, "hardware_service", None)
        # Threshold set to 16GB. Triggers early enough to leave headroom for model swaps.
        # (Whisper ~150MB + Ollama ~6GB + OS overhead = guard needs to fire well before 24GB)
        self.vram_threshold_mb = 16384
        self.guard_loop.start()

    def cog_unload(self):
        """Cleanly cancel the loop when the cog is unloaded."""
        self.guard_loop.cancel()

    def _get_vram_usage_mb(self) -> int:
        if self.hardware_service:
            return self.hardware_service.get_vram_usage_mb()
        return 0

    # --- NEW: MANUAL CLEAR ADDITION ---
    async def force_clear(self):
        """Manual trigger to dump VRAM across all local services."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return self._get_vram_usage_mb()

    # --- NEW: COMMAND FOR STREAMING FEEDBACK ---
    @commands.command(name="vram")
    @commands.is_owner()
    async def vram_status(self, ctx):
        """Check current 3090 Ti VRAM and hardware status."""
        usage = self._get_vram_usage_mb()
        status = "🟢 HEALTHY" if usage < self.vram_threshold_mb else "🔴 HIGH USAGE"
        await ctx.send(f"📊 **3090 Ti VRAM Status:** `{usage}MB / 24576MB` | Status: {status}")

    @tasks.loop(minutes=5)
    async def guard_loop(self):
        """
        Periodically checks VRAM levels.
        If high usage is detected while the bot is IDLE, it performs a cache purge.
        """
        # CRITICAL: Check the safety counter from AgentDispatcher
        if getattr(self.bot, "generating_count", 0) > 0:
            # Do not cross wires while a model is actively rendering
            return

        current_usage = await asyncio.to_thread(self._get_vram_usage_mb)
        
        if current_usage > self.vram_threshold_mb:
            logger.info("[VRAM GUARD] High idle usage detected: %sMB. Initializing stabilizer...", current_usage)
            
            # 1. Python Garbage Collection
            gc.collect()
            
            # 2. PyTorch CUDA Cache Clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            new_usage = await asyncio.to_thread(self._get_vram_usage_mb)
            freed = current_usage - new_usage
            logger.info("[VRAM GUARD] Stabilization complete. Freed %sMB. Current: %sMB.", freed, new_usage)

            # Notify bot owner via DM
            try:
                app_info = await self.bot.application_info()
                owner = app_info.owner
                if owner:
                    await owner.send(
                        f"⚠️ **VRAM Guard triggered**\n"
                        f"Idle VRAM was `{current_usage}MB`, freed `{freed}MB`, now `{new_usage}MB`."
                    )
            except Exception as exc:
                logger.debug("[VRAM GUARD] Could not DM owner: %s", exc)

    @guard_loop.before_loop
    async def before_guard_loop(self):
        """Wait until the bot is fully logged in before starting the monitor."""
        await self.bot.wait_until_ready()
        logger.info("[VRAM GUARD] 3090 Ti Hardware Monitor active.")

async def setup(bot):
    await bot.add_cog(VRAMGuard(bot))