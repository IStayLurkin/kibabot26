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
        import psutil
        import subprocess

        # First try graceful keep_alive=0
        model_name = self._get_active_ollama_model()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    logger.info("[wan] Ollama unload request: %s", resp.status)
        except Exception as e:
            logger.warning("[wan] Ollama unload request failed: %s", e)

        await asyncio.sleep(5)

        # Check if memory freed — if not, force-restart Ollama service
        vram_free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024 / 1024
        ram_free_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
        logger.info("[wan] After soft unload — VRAM free: %.0fMB | RAM free: %.1fGB", vram_free_mb, ram_free_gb)

        if vram_free_mb < 18000 or ram_free_gb < 30:
            logger.info("[wan] Memory not cleared — force-stopping Ollama service...")
            try:
                subprocess.run(["ollama", "stop"], capture_output=True, timeout=15)
            except Exception:
                pass
            # Kill any remaining ollama processes
            for proc in psutil.process_iter(["name", "pid"]):
                try:
                    if "ollama" in proc.info["name"].lower():
                        proc.kill()
                        logger.info("[wan] Killed ollama process pid=%s", proc.info["pid"])
                except Exception:
                    pass

        # Wait for both VRAM and system RAM to free up
        for i in range(60):
            await asyncio.sleep(2)
            torch.cuda.empty_cache()
            gc.collect()
            vram_free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024 / 1024
            ram_free_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
            logger.info("[wan] VRAM free: %.0fMB | RAM free: %.1fGB", vram_free_mb, ram_free_gb)
            if vram_free_mb > 18000 and ram_free_gb > 30:
                logger.info("[wan] Memory cleared, ready to load.")
                break
        else:
            logger.warning("[wan] Memory did not fully clear after 120s — proceeding anyway.")

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
            WAN_REPO, torch_dtype=torch.float16, local_files_only=True
        )
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        self.pipeline.enable_sequential_cpu_offload()
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
        frames_np = [(np.array(f) * 255).clip(0, 255).astype(np.uint8) for f in frames]
        imageio.mimwrite(filepath, frames_np, fps=16, codec="libx264")
        logger.info("[wan] Video saved: %s (%.1f MB)", filepath, Path(filepath).stat().st_size / 1024 / 1024)
        return filepath

    async def _restart_ollama(self):
        import subprocess
        import psutil
        try:
            logger.info("[wan] Restarting Ollama...")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Wait for Ollama to be ready
            for _ in range(30):
                await asyncio.sleep(2)
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                            if resp.status == 200:
                                logger.info("[wan] Ollama back online.")
                                return
                except Exception:
                    pass
            logger.warning("[wan] Ollama did not come back online after 60s.")
        except Exception as e:
            logger.warning("[wan] Failed to restart Ollama: %s", e)

    async def generate(self, prompt: str, callback) -> Optional[str]:
        await self._unload_ollama()
        filename = f"wan_{int(time.time())}.mp4"
        filepath = str(OUTPUT_DIR / filename)
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                HEAVY_EXECUTOR, self._generate_sync, prompt, filepath, callback
            )
            return result
        except Exception as e:
            logger.exception("[wan] Generation failed: %s", e)
            return None
        finally:
            self._purge_vram()
            await self._restart_ollama()
