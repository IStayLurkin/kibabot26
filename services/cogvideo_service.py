from __future__ import annotations

import gc
import os
import time
import asyncio
import torch
from pathlib import Path
from typing import Optional

from core.executors import HEAVY_EXECUTOR
from core.logging_config import get_logger
from services.hardware_service import HardwareService

logger = get_logger(__name__)

try:
    from diffusers import CogVideoXPipeline
    import imageio
except ImportError:
    CogVideoXPipeline = None
    imageio = None

COGVIDEO_2B_REPO = "THUDM/CogVideoX-2b"
COGVIDEO_5B_REPO = "THUDM/CogVideoX-5b"
OUTPUT_DIR = Path("outputs/videos")


class CogVideoService:
    def __init__(self):
        self.pipeline = None
        self.current_model: str | None = None
        self._hardware = HardwareService()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _get_vram_usage(self) -> int:
        return self._hardware.get_vram_usage_mb()

    def _purge_vram(self):
        if self.pipeline is not None:
            logger.debug("Purging CogVideoX from VRAM...")
            self.pipeline = None
            self.current_model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _load(self, model_size: str):
        repo = COGVIDEO_2B_REPO if model_size == "2b" else COGVIDEO_5B_REPO
        if self.current_model == model_size and self.pipeline is not None:
            return
        self._purge_vram()
        logger.info("[cogvideo] Loading CogVideoX-%s...", model_size)
        self.pipeline = CogVideoXPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        self.pipeline.enable_sequential_cpu_offload()
        self.current_model = model_size
        logger.info("[cogvideo] CogVideoX-%s loaded.", model_size)

    def _generate_sync(self, model_size: str, prompt: str, filepath: str, callback) -> str:
        self._load(model_size)
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
            width=720,
            callback_on_step_end=step_callback,
        )

        # Export frames to MP4
        frames = output.frames[0]  # list of PIL images
        frames_np = [frame if hasattr(frame, 'shape') else __import__('numpy').array(frame) for frame in frames]
        imageio.mimwrite(filepath, frames_np, fps=8, codec="libx264")
        return filepath

    async def generate(self, model_size: str, prompt: str, callback) -> Optional[str]:
        if CogVideoXPipeline is None or imageio is None:
            logger.error("[cogvideo] CogVideoXPipeline or imageio not available.")
            return None
        filename = f"cogvideo_{model_size}_{int(time.time())}.mp4"
        filepath = str(OUTPUT_DIR / filename)
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                HEAVY_EXECUTOR, self._generate_sync, model_size, prompt, filepath, callback
            )
        except Exception as e:
            logger.error("[cogvideo] Generation failed: %s", e)
            self._purge_vram()
            return None
