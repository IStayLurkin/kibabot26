from __future__ import annotations

import gc
import os
import time
import asyncio
from pathlib import Path
from typing import Optional

import torch

from core.executors import HEAVY_EXECUTOR
from core.logging_config import get_logger
from services.hardware_service import HardwareService

logger = get_logger(__name__)

WAN_1_3B_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUTPUT_DIR = Path("J:/aistorage/generated_media/videos")


class WanFastService:
    def __init__(self):
        self.pipeline = None
        self._hardware = HardwareService()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _get_vram_usage(self) -> int:
        return self._hardware.get_vram_usage_mb()

    def _purge_vram(self):
        if self.pipeline is not None:
            logger.debug("[wan_fast] Purging from VRAM...")
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _load(self):
        if self.pipeline is not None:
            return
        try:
            from diffusers import WanPipeline as _WanPipeline
        except ImportError as exc:
            raise RuntimeError("diffusers is not installed — cannot load Wan2.1-1.3B") from exc
        logger.info("[wan_fast] Loading Wan2.1-T2V-1.3B-Diffusers...")
        self.pipeline = _WanPipeline.from_pretrained(
            WAN_1_3B_REPO, torch_dtype=torch.float16, local_files_only=True
        )
        self.pipeline.to("cuda")
        logger.info("[wan_fast] Wan2.1-1.3B loaded.")

    def _generate_sync(self, prompt: str, filepath: str, callback) -> str:
        self._load()
        steps = 50
        last_update = [0.0]

        logger.info("[wan_fast] Starting inference — prompt: %s", prompt[:80])
        output = self.pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            num_frames=81,
            height=480,
            width=832,
        )
        logger.info("[wan_fast] Inference complete.")
        try:
            import imageio
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("imageio or numpy not installed — cannot save video") from exc

        frames = output.frames[0]
        logger.info("[wan_fast] Generation complete, saving %d frames to %s", len(frames), filepath)
        frames_np = [(np.array(f) * 255).clip(0, 255).astype(np.uint8) for f in frames]
        imageio.mimwrite(filepath, frames_np, fps=16, codec="libx264")
        logger.info("[wan_fast] Video saved: %s (%.1f MB)", filepath, Path(filepath).stat().st_size / 1024 / 1024)
        return filepath

    async def _kill_ollama(self):
        import psutil
        import subprocess
        try:
            subprocess.run(["ollama", "stop"], capture_output=True, timeout=10)
        except Exception:
            pass
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if "ollama" in proc.info["name"].lower():
                    proc.kill()
                    logger.info("[wan_fast] Killed ollama pid=%s", proc.info["pid"])
            except Exception:
                pass
        # Wait for RAM to free
        import psutil as _psutil
        for _ in range(30):
            await asyncio.sleep(2)
            ram_free_gb = _psutil.virtual_memory().available / 1024**3
            logger.info("[wan_fast] RAM free: %.1fGB", ram_free_gb)
            if ram_free_gb > 30:
                break

    async def _restart_ollama(self):
        import subprocess
        import aiohttp
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(30):
                await asyncio.sleep(2)
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                            if resp.status == 200:
                                logger.info("[wan_fast] Ollama back online.")
                                return
                except Exception:
                    pass
            logger.warning("[wan_fast] Ollama did not come back online.")
        except Exception as e:
            logger.warning("[wan_fast] Failed to restart Ollama: %s", e)

    async def generate(self, prompt: str, callback) -> Optional[str]:
        filename = f"wan_fast_{int(time.time())}.mp4"
        filepath = str(OUTPUT_DIR / filename)
        await self._kill_ollama()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                HEAVY_EXECUTOR, self._generate_sync, prompt, filepath, callback
            )
            return result
        except Exception as e:
            logger.exception("[wan_fast] Generation failed: %s", e)
            return None
        finally:
            self._purge_vram()
            await self._restart_ollama()
