from __future__ import annotations

import gc
import time
import asyncio
import aiohttp
import torch
from pathlib import Path
from typing import Optional

from core.executors import HEAVY_EXECUTOR
from core.logging_config import get_logger
from core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from services.hardware_service import HardwareService

logger = get_logger(__name__)

try:
    from diffusers import WanPipeline
    import imageio
except ImportError:
    WanPipeline = None
    imageio = None

WAN_REPO = "Wan-AI/Wan2.1-T2V-14B"
OUTPUT_DIR = Path("outputs/videos")


class WanService:
    def __init__(self, runtime_service=None):
        self.pipeline = None
        self.runtime_service = runtime_service
        self._hardware = HardwareService()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _get_vram_usage(self) -> int:
        return self._hardware.get_vram_usage_mb()

    def _get_active_ollama_model(self) -> str:
        if self.runtime_service is not None:
            try:
                name = self.runtime_service.get_active_llm_model()
                if name:
                    return name
            except Exception:
                pass
        return OLLAMA_MODEL

    async def _unload_ollama(self):
        model_name = self._get_active_ollama_model()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json={"model": model_name, "keep_alive": 0}
                ) as resp:
                    if resp.status == 200:
                        logger.debug("[wan] Ollama ejected from VRAM.")
            await asyncio.sleep(1)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.debug("[wan] Ollama unload failed: %s", e)

    def _purge_vram(self):
        if self.pipeline is not None:
            logger.debug("Purging Wan from VRAM...")
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _load(self):
        if self.pipeline is not None:
            return
        logger.info("[wan] Loading Wan2.1-T2V-14B...")
        self.pipeline = WanPipeline.from_pretrained(WAN_REPO, torch_dtype=torch.bfloat16)
        self.pipeline.enable_model_cpu_offload()
        logger.info("[wan] Wan2.1 loaded.")

    def _generate_sync(self, prompt: str, filepath: str, callback) -> str:
        self._load()
        steps = 50
        last_update = [0.0]

        def step_callback(pipe, step, timestep, kwargs):
            now = time.time()
            if callback and (now - last_update[0] >= 3.0 or step == steps - 1):
                percent = int((step / steps) * 100)
                vram = round(self._get_vram_usage() / 1024, 1)
                callback(percent, vram)
                last_update[0] = now
            return kwargs

        output = self.pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            num_frames=49,
            height=480,
            width=832,
            callback_on_step_end=step_callback,
        )
        import numpy as np
        frames = output.frames[0]
        frames_np = [np.array(f) for f in frames]
        imageio.mimwrite(filepath, frames_np, fps=16, codec="libx264")
        return filepath

    async def generate(self, prompt: str, callback) -> Optional[str]:
        if WanPipeline is None or imageio is None:
            logger.error("[wan] WanPipeline or imageio not available.")
            return None
        await self._unload_ollama()
        filename = f"wan_{int(time.time())}.mp4"
        filepath = str(OUTPUT_DIR / filename)
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                HEAVY_EXECUTOR, self._generate_sync, prompt, filepath, callback
            )
        except Exception as e:
            logger.error("[wan] Generation failed: %s", e)
            self._purge_vram()
            return None
