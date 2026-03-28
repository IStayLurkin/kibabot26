from __future__ import annotations

import gc
import os
import re
import time
import asyncio
from pathlib import Path
from typing import Optional

import torch

from core.executors import HEAVY_EXECUTOR
from core.logging_config import get_logger

logger = get_logger(__name__)

FOUNDATION_REPO = "J:/aistorage/huggingface_cache/hub/Foundation-1"
FOUNDATION_CKPT = "J:/aistorage/huggingface_cache/hub/Foundation-1/Foundation_1.safetensors"
FOUNDATION_CONFIG = "J:/aistorage/huggingface_cache/hub/Foundation-1/model_config.json"
OUTPUT_DIR = Path("J:/aistorage/generated_media/music")

# BPM * bars / 60 * 4 beats per bar = duration in seconds
# e.g. 4 bars at 120 BPM = 4 * 4 / 120 * 60 = 8s
DEFAULT_BPM = 120
DEFAULT_BARS = 4
DEFAULT_STEPS = 100
DEFAULT_CFG = 7.0


def _parse_bpm(prompt: str) -> int:
    m = re.search(r"\b(\d{2,3})\s*bpm\b", prompt, re.IGNORECASE)
    return int(m.group(1)) if m else DEFAULT_BPM


def _parse_bars(prompt: str) -> int:
    m = re.search(r"\b(\d+)\s*bar\b", prompt, re.IGNORECASE)
    return int(m.group(1)) if m else DEFAULT_BARS


def _bars_to_seconds(bars: int, bpm: int) -> float:
    return (bars * 4 / bpm) * 60


class FoundationService:
    def __init__(self):
        self.model = None
        self.model_config = None
        self._lock = asyncio.Lock()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if self.model is not None:
            return
        import json
        from stable_audio_tools import get_pretrained_model
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        logger.info("[foundation] Loading Foundation-1...")
        with open(FOUNDATION_CONFIG) as f:
            self.model_config = json.load(f)

        from stable_audio_tools.models.factory import create_model_from_config
        self.model = create_model_from_config(self.model_config)
        state_dict = load_ckpt_state_dict(FOUNDATION_CKPT)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to("cuda").eval()
        logger.info("[foundation] Loaded.")

    def _unload(self):
        if self.model is not None:
            self.model = None
            self.model_config = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[foundation] Unloaded from VRAM.")

    def _generate_sync(self, prompt: str, filepath: str) -> str:
        self._load()
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        bpm = _parse_bpm(prompt)
        bars = _parse_bars(prompt)
        duration = _bars_to_seconds(bars, bpm)
        sample_rate = self.model_config.get("sample_rate", 44100)
        sample_size = int(duration * sample_rate)

        logger.info("[foundation] Generating — prompt: %s | bpm=%d bars=%d duration=%.1fs", prompt[:80], bpm, bars, duration)

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration,
        }]

        import numpy as np
        seed = int(np.random.randint(0, 2**31 - 1))

        with torch.no_grad():
            output = generate_diffusion_cond(
                self.model,
                steps=DEFAULT_STEPS,
                cfg_scale=DEFAULT_CFG,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device="cuda",
                seed=seed,
            )

        # output shape: [batch, channels, samples]
        audio = output[0].cpu()

        import torchaudio
        torchaudio.save(filepath, audio, sample_rate)
        logger.info("[foundation] Saved: %s", filepath)
        return filepath

    async def generate(self, prompt: str) -> Optional[str]:
        filename = f"foundation_{int(time.time())}.wav"
        filepath = str(OUTPUT_DIR / filename)
        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    HEAVY_EXECUTOR,
                    self._generate_sync, prompt, filepath
                )
                return result
            except Exception as e:
                logger.exception("[foundation] Generation failed: %s", e)
                return None
            finally:
                gc.collect()
                torch.cuda.empty_cache()
