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

ZIMAGE_REPO = "J:/aistorage/huggingface_cache/hub/Z-Image-Turbo"
OUTPUT_DIR = "J:/aistorage/generated_media/images"


class ZImageService:
    def __init__(self):
        self.pipeline = None
        self._lock = asyncio.Lock()
        self.last_update_time: float = 0.0
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _load(self):
        if self.pipeline is not None:
            return
        from diffusers import ZImagePipeline
        logger.info("[zimage] Loading Z-Image-Turbo...")
        self.pipeline = ZImagePipeline.from_pretrained(
            ZIMAGE_REPO,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        logger.info("[zimage] Loaded.")

    def _unload(self):
        if self.pipeline is not None:
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[zimage] Unloaded from VRAM.")

    def _generate_sync(self, prompt: str, filepath: str, callback, loop) -> str:
        self._load()

        steps = 15

        def pipe_callback(pipe, step, timestep, callback_kwargs):
            if callback:
                now = time.time()
                if now - self.last_update_time >= 1.0 or step == steps:
                    percent = int((step / steps) * 100)
                    asyncio.run_coroutine_threadsafe(callback(percent), loop)
                    self.last_update_time = now
            return callback_kwargs

        logger.info("[zimage] Generating — prompt: %s", prompt[:80])
        image = self.pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            callback_on_step_end=pipe_callback,
        ).images[0]

        image.save(filepath)
        logger.info("[zimage] Saved: %s", filepath)
        return filepath

    async def generate(self, prompt: str, callback=None) -> Optional[str]:
        filename = f"zimage_{int(time.time())}.png"
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
                logger.exception("[zimage] Generation failed: %s", e)
                return None
            finally:
                gc.collect()
                torch.cuda.empty_cache()
