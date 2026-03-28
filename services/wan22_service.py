from __future__ import annotations

import gc
import os
import time
import asyncio
import subprocess
import psutil
import aiohttp
from pathlib import Path
from typing import Optional

import torch

from core.executors import HEAVY_EXECUTOR
from core.logging_config import get_logger
from core.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = get_logger(__name__)

WAN22_REPO = "J:/aistorage/Wan2.2"
WAN22_CKPT = "J:/aistorage/huggingface_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
OUTPUT_DIR = Path("J:/aistorage/generated_media/videos")
VENV_PYTHON = str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe")


class Wan22Service:
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async def _kill_ollama(self):
        try:
            subprocess.run(["ollama", "stop"], capture_output=True, timeout=10)
        except Exception:
            pass
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if "ollama" in proc.info["name"].lower():
                    proc.kill()
                    logger.info("[wan22] Killed ollama pid=%s", proc.info["pid"])
            except Exception:
                pass
        for _ in range(30):
            await asyncio.sleep(2)
            ram_free_gb = psutil.virtual_memory().available / 1024**3
            vram_free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024 / 1024
            logger.info("[wan22] RAM free: %.1fGB | VRAM free: %.0fMB", ram_free_gb, vram_free_mb)
            if ram_free_gb > 30 and vram_free_mb > 18000:
                logger.info("[wan22] Memory cleared, ready.")
                break
        else:
            logger.warning("[wan22] Memory did not fully clear — proceeding anyway.")

    async def _restart_ollama(self):
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(30):
                await asyncio.sleep(2)
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                            if resp.status == 200:
                                logger.info("[wan22] Ollama back online.")
                                return
                except Exception:
                    pass
            logger.warning("[wan22] Ollama did not come back online.")
        except Exception as e:
            logger.warning("[wan22] Failed to restart Ollama: %s", e)

    def _generate_sync(self, prompt: str, filepath: str, callback) -> str:
        if not Path(WAN22_REPO).exists():
            raise RuntimeError(f"Wan2.2 repo not found at {WAN22_REPO}")
        if not Path(WAN22_CKPT).exists():
            raise RuntimeError(f"Wan2.2 checkpoint not found at {WAN22_CKPT}")
        if not Path(VENV_PYTHON).exists():
            raise RuntimeError(f"venv python not found at {VENV_PYTHON}")

        cmd = [
            VENV_PYTHON, "generate.py",
            "--task", "ti2v-5B",
            "--ckpt_dir", WAN22_CKPT,
            "--size", "1280*704",
            "--frame_num", "81",
            "--offload_model", "True",
            "--t5_cpu",
            "--save_file", filepath,
            "--prompt", prompt,
            "--sample_steps", "50",
        ]

        logger.info("[wan22] Starting generation — prompt: %s", prompt[:80])
        if callback:
            callback(0, 0.0)

        proc = subprocess.Popen(
            cmd,
            cwd=WAN22_REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        import re
        start = time.time()
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            logger.info("[wan22] %s", line)
            # Parse tqdm diffusion steps only: e.g. " 42%|████  | 21/50 [01:23<...]"
            # Match only when total == sample_steps (50), not shard loading (3) or other bars
            m = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)", line)
            if m and callback and int(m.group(3)) == 50:
                percent = int(m.group(1))
                callback(percent, 0.0)

        proc.wait(timeout=1800)
        elapsed = int(time.time() - start)

        if proc.returncode != 0:
            raise RuntimeError(f"Wan2.2 generation failed (exit {proc.returncode}) after {elapsed}s")

        logger.info("[wan22] Generation finished in %ds.", elapsed)
        if not Path(filepath).exists():
            raise RuntimeError("Wan2.2 generate.py succeeded but output file not found")

        size_mb = Path(filepath).stat().st_size / 1024 / 1024
        logger.info("[wan22] Video saved: %s (%.1fMB)", filepath, size_mb)
        if callback:
            callback(100, 0.0)
        return filepath

    async def generate(self, prompt: str, callback) -> Optional[str]:
        filename = f"wan22_{int(time.time())}.mp4"
        filepath = str(OUTPUT_DIR / filename)
        await self._kill_ollama()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                HEAVY_EXECUTOR, self._generate_sync, prompt, filepath, callback
            )
            return result
        except Exception as e:
            logger.exception("[wan22] Generation failed: %s", e)
            return None
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            await self._restart_ollama()
