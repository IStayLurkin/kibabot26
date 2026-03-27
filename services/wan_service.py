from __future__ import annotations

import gc
import os
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

# WanPipeline and imageio are imported lazily inside _load() to avoid triggering
# HuggingFace snapshot checks (and 37GB downloads) at bot startup.

WAN_REPO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
OUTPUT_DIR = Path("J:/aistorage/generated_media/videos")


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
                    json={"model": model_name, "keep_alive": 0},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        logger.debug("[wan] Ollama ejected from VRAM.")
            # Wait for Ollama to actually release CUDA memory
            for _ in range(10):
                await asyncio.sleep(1)
                torch.cuda.empty_cache()
                gc.collect()
                free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024 / 1024
                logger.debug("[wan] VRAM free after unload: %.0fMB", free_mb)
                if free_mb > 8000:
                    break
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
        try:
            from diffusers import WanPipeline as _WanPipeline
        except ImportError as exc:
            raise RuntimeError("diffusers is not installed — cannot load Wan2.1") from exc
        logger.info("[wan] Loading Wan2.1-T2V-14B-Diffusers (720p, 81 frames)...")
        self.pipeline = _WanPipeline.from_pretrained(
            WAN_REPO, torch_dtype=torch.bfloat16, local_files_only=True
        )
        # Keep text encoder on CPU to avoid OOM — it gets offloaded per-forward automatically
        self.pipeline.text_encoder.to("cpu")
        self.pipeline.enable_model_cpu_offload()
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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
            num_frames=81,
            height=720,
            width=1280,
            callback_on_step_end=step_callback,
        )
        try:
            import imageio
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("imageio or numpy not installed — cannot save video") from exc
        frames = output.frames[0]
        logger.info("[wan] Generation complete, saving %d frames to %s", len(frames), filepath)
        frames_np = [np.array(f) for f in frames]
        imageio.mimwrite(filepath, frames_np, fps=16, codec="libx264")
        logger.info("[wan] Video saved: %s (%.1f MB)", filepath, Path(filepath).stat().st_size / 1024 / 1024)
        return filepath

    async def generate(self, prompt: str, callback) -> Optional[str]:
        await self._unload_ollama()
        filename = f"wan_{int(time.time())}.mp4"
        filepath = str(OUTPUT_DIR / filename)
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                HEAVY_EXECUTOR, self._generate_sync, prompt, filepath, callback
            )
        except Exception as e:
            logger.exception("[wan] Generation failed: %s", e)
            self._purge_vram()
            return None
