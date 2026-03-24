from __future__ import annotations

import gc
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
    from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
    from diffusers.utils import export_to_video
    import imageio
except ImportError:
    AnimateDiffPipeline = None
    MotionAdapter = None
    EulerDiscreteScheduler = None
    export_to_video = None
    imageio = None

MOTION_ADAPTER_REPO = "guoyww/animatediff-motion-adapter-v1-5-2"
BASE_MODEL_REPO = "SG161222/Realistic_Vision_V5.1_noVAE"
OUTPUT_DIR = Path("outputs/videos")


class AnimateDiffService:
    def __init__(self):
        self.pipeline = None
        self._hardware = HardwareService()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _get_vram_usage(self) -> int:
        return self._hardware.get_vram_usage_mb()

    def _purge_vram(self):
        if self.pipeline is not None:
            logger.debug("Purging AnimateDiff from VRAM...")
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _load(self):
        if self.pipeline is not None:
            return
        logger.info("[animatediff] Loading AnimateDiff pipeline...")
        adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_REPO, torch_dtype=torch.float16)
        self.pipeline = AnimateDiffPipeline.from_pretrained(
            BASE_MODEL_REPO,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, beta_schedule="linear"
        )
        self.pipeline.enable_vae_slicing()
        logger.info("[animatediff] Pipeline loaded.")

    def _generate_sync(self, prompt: str, filepath: str, callback) -> str:
        self._load()
        steps = 25
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
            negative_prompt="bad quality, worse quality, blurry",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=steps,
            height=512,
            width=512,
            callback_on_step_end=step_callback,
        )
        frames = output.frames[0]
        import numpy as np
        frames_np = [np.array(f) for f in frames]
        imageio.mimwrite(filepath, frames_np, fps=8, codec="libx264")
        return filepath

    async def generate(self, prompt: str, callback) -> Optional[str]:
        if AnimateDiffPipeline is None or imageio is None:
            logger.error("[animatediff] AnimateDiffPipeline or imageio not available.")
            return None
        filename = f"animatediff_{int(time.time())}.mp4"
        filepath = str(OUTPUT_DIR / filename)
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                HEAVY_EXECUTOR, self._generate_sync, prompt, filepath, callback
            )
        except Exception as e:
            logger.error("[animatediff] Generation failed: %s", e)
            self._purge_vram()
            return None
