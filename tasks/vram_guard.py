import torch
import gc
import asyncio
import subprocess
from discord.ext import tasks, commands

class VRAMGuard(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        # Threshold set to 21GB. If idle VRAM exceeds this, we flush.
        self.vram_threshold_mb = 21504 
        self.guard_loop.start()

    def cog_unload(self):
        """Cleanly cancel the loop when the cog is unloaded."""
        self.guard_loop.cancel()

    def _get_vram_usage_mb(self) -> int:
        """Direct hardware query for current 3090 Ti memory usage."""
        try:
            cmd = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
            result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            return int(result)
        except Exception as e:
            print(f"[VRAM Guard Error] Could not query nvidia-smi: {e}")
            return 0

    @tasks.loop(minutes=5)
    async def guard_loop(self):
        """
        Periodically checks VRAM levels. 
        If high usage is detected while the bot is IDLE, it performs a cache purge.
        """
        # CRITICAL: Check the safety flag from AgentDispatcher
        is_busy = getattr(self.bot, "is_generating", False)
        
        if is_busy:
            # Do not cross wires while a model is actively rendering
            return

        current_usage = self._get_vram_usage_mb()
        
        if current_usage > self.vram_threshold_mb:
            print(f"[VRAM GUARD] High idle usage detected: {current_usage}MB. Initializing stabilizer...")
            
            # 1. Python Garbage Collection
            gc.collect()
            
            # 2. PyTorch CUDA Cache Clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            new_usage = self._get_vram_usage_mb()
            freed = current_usage - new_usage
            print(f"[VRAM GUARD] Stabilization complete. Freed {freed}MB. Current: {new_usage}MB.")

    @guard_loop.before_loop
    async def before_guard_loop(self):
        """Wait until the bot is fully logged in before starting the monitor."""
        await self.bot.wait_until_ready()
        print("[VRAM GUARD] 3090 Ti Hardware Monitor active.")

async def setup(bot):
    await bot.add_cog(VRAMGuard(bot))