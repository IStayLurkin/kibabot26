from __future__ import annotations

import gc
import os
import time
import asyncio
from typing import Optional

import torch

from core.executors import HEAVY_EXECUTOR
from core.logging_config import get_logger

logger = get_logger(__name__)

FLUX2_KLEIN_REPO = "J:/aistorage/huggingface_cache/hub/FLUX2-klein-kv"
OUTPUT_DIR = "J:/aistorage/generated_media/images"


class Flux2KleinService:
    def __init__(self):
        self.pipeline = None
        self._lock = asyncio.Lock()
        self.last_update_time: float = 0.0
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _load(self):
        if self.pipeline is not None:
            return
        from diffusers import Flux2KleinPipeline
        logger.info("[flux2klein] Loading FLUX.2 Klein (~17GB FP8)...")
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            FLUX2_KLEIN_REPO,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        logger.info("[flux2klein] Loaded.")

    def _unload(self):
        if self.pipeline is not None:
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[flux2klein] Unloaded from VRAM.")

    def _generate_sync(self, prompt: str, filepath: str, callback, loop) -> str:
        self._load()

        steps = 4  # distilled model — 4 steps is optimal

        def pipe_callback(pipe, step, timestep, callback_kwargs):
            if callback:
                now = time.time()
                if now - self.last_update_time >= 0.5 or step == steps:
                    percent = int((step / steps) * 100)
                    asyncio.run_coroutine_threadsafe(callback(percent), loop)
                    self.last_update_time = now
            return callback_kwargs

        logger.info("[flux2klein] Generating — prompt: %s", prompt[:80])
        image = self.pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            height=1024,
            width=1024,
            callback_on_step_end=pipe_callback,
        ).images[0]

        image.save(filepath)
        logger.info("[flux2klein] Saved: %s", filepath)
        return filepath

    async def generate(self, prompt: str, callback=None) -> Optional[str]:
        filename = f"flux2klein_{int(time.time())}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    HEAVY_EXECUTOR,
                    self._generate_sync, prompt, filepath, callback, loop
                )
                return result
            except Exception as e:
                logger.exception("[flux2klein] Generation failed: %s", e)
                return None
            finally:
                gc.collect()
                torch.cuda.empty_cache()
