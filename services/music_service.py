from __future__ import annotations

import asyncio
import gc
import os
import time
import uuid
import torch
import aiohttp
from pathlib import Path
from typing import Optional

# 2026 Local Foundation Model Imports
try:
    from diffusers import StableAudioProjectionPipeline
    # Note: YuE typically utilizes a custom transformer/diffusers hybrid pipeline
    # We use a placeholder for the specific YuE loader class here
    from core.model_loaders import YueFoundationModel 
except ImportError:
    StableAudioProjectionPipeline = None
    YueFoundationModel = None

from core.config import (
    MUSIC_DEFAULT_QUALITY,
    MUSIC_REQUEST_TIMEOUT_SECONDS,
)
from core.feature_flags import MEDIA_OUTPUT_DIR

class MusicService:
    def __init__(self, performance_tracker=None) -> None:
        self.performance_tracker = performance_tracker
        self.output_dir = Path(MEDIA_OUTPUT_DIR) / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None  # Lazy-loaded foundation model
        self.active_model_type = None # Track what's in VRAM

    async def _unload_ollama(self):
        """Standard 3090 Ti VRAM clearance for heavy foundation models."""
        try:
            print("[DEBUG] Clearing VRAM for Studio Audio Generation...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json={"model": "qwen3-coder:7b", "keep_alive": 0}
                ) as resp:
                    if resp.status == 200:
                        print("[DEBUG] Qwen3 ejected from VRAM.")
            await asyncio.sleep(1)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"[DEBUG] Unload failed: {e}")

    async def generate_melody(self, prompt: str) -> str:
        """Utilizes Stable Audio Open for high-fidelity loops/melodies."""
        await self._unload_ollama()
        
        filename = f"melody_{int(time.time())}.wav"
        path = self.output_dir / filename

        try:
            return await asyncio.to_thread(self._generate_melody_local, prompt, str(path))
        except Exception as e:
            print(f"[ERROR] Melody Gen Failed: {e}")
            return ""

    async def generate_song_clip(
        self,
        vibe: str,
        bpm: int,
        voice_style: str,
        vocal_mode: str,
        lyrics: str = ""
    ) -> str:
        """Utilizes YuE Foundation Model for human-level studio vocals."""
        await self._unload_ollama()
        
        filename = f"kiba_studio_{int(time.time())}.mp3"
        path = self.output_dir / filename

        prompt = f"{vibe}, {voice_style} vocals, {vocal_mode}, {bpm} BPM. {lyrics}"

        try:
            return await asyncio.to_thread(self._generate_yue_studio, prompt, str(path))
        except Exception as e:
            print(f"[ERROR] YuE Studio Gen Failed: {e}")
            return ""
        

    def update_studio_settings(self, bpm: int = None, voice: str = None, mode: str = None):
            """Updates the local studio configuration for YuE synthesis."""
            if bpm:
                self.bpm = max(60, min(200, bpm))
            if voice:
                self.voice_style = voice.strip().lower()
            if mode:
                self.vocal_mode = mode.strip().lower()
            
            print(f"[DEBUG] Studio Config Updated: {self.bpm} BPM | {self.voice_style} | {self.vocal_mode}")

    def _generate_melody_local(self, prompt: str, filepath: str) -> str:
        """Sync worker for Stable Audio Open."""
        if self.active_model_type != "stable-audio":
            self.pipeline = StableAudioProjectionPipeline.from_pretrained(
                "stabilityai/stable-audio-open-1.0", 
                torch_dtype=torch.float16
            )
            self.pipeline.to(self.device)
            self.active_model_type = "stable-audio"

        # Generate 15-second high-quality loop
        audio = self.pipeline(
            prompt, 
            steps=100, 
            seconds_total=15
        ).audios[0]
        
        # Save logic (assumes standard audio saving utility)
        self._save_audio(audio, filepath)
        return filepath

    def _generate_yue_studio(self, prompt: str, filepath: str) -> str:
        """Sync worker for YuE Studio Vocals (VRAM heavy)."""
        if self.active_model_type != "yue":
            print("[DEBUG] Loading YuE Foundation Model (32B-Audio)...")
            # 4-bit quantization is often needed for YuE on 24GB cards
            self.pipeline = YueFoundationModel.from_pretrained(
                "m-a-p/YuE-s1-7B-anneal-en-cot",
                load_in_4bit=True,
                torch_dtype=torch.bfloat16
            )
            # Use Sequential Offloading just like FLUX.2
            self.pipeline.enable_sequential_cpu_offload()
            self.active_model_type = "yue"

        # YuE generates realistic lyrics-to-singing
        audio_stream = self.pipeline.generate_full_song(
            prompt,
            num_steps=50,
            guidance_scale=5.0
        )
        
        audio_stream.save(filepath)
        return filepath

    def _save_audio(self, audio_data, filepath: str):
        """Utility to write raw tensors to wav/mp3."""
        import scipy.io.wavfile as wavfile
        wavfile.write(filepath, 44100, audio_data.cpu().numpy())

    def clear_vram(self):
        """Force ejection of all audio models."""
        self.pipeline = None
        self.active_model_type = None
        torch.cuda.empty_cache()
        gc.collect()