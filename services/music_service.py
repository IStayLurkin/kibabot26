from __future__ import annotations

import asyncio
import gc
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import torch
import aiohttp
from pathlib import Path
from typing import Optional

try:
    from diffusers import StableAudioPipeline
except ImportError:
    StableAudioPipeline = None

from core.config import (
    MUSIC_DEFAULT_QUALITY,
    MUSIC_REQUEST_TIMEOUT_SECONDS,
    OLLAMA_MODEL,
    YUE_REPO_PATH,
)
from core.executors import HEAVY_EXECUTOR
from core.feature_flags import MEDIA_OUTPUT_DIR
from core.logging_config import get_logger

logger = get_logger(__name__)

YUE_STAGE1_MODEL = "m-a-p/YuE-s1-7B-anneal-en-cot"
YUE_STAGE2_MODEL = "m-a-p/YuE-s2-1B-general"


class MusicService:
    def __init__(self, performance_tracker=None, runtime_service=None) -> None:
        self.performance_tracker = performance_tracker
        self.runtime_service = runtime_service
        self.output_dir = Path(MEDIA_OUTPUT_DIR) / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.active_model_type = None
        self.bpm = 120
        self.voice_style = "studio"
        self.vocal_mode = "lyrics"

    def _get_active_ollama_model(self) -> str:
        """Return the live runtime model name, falling back to config."""
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
            logger.debug("Clearing VRAM for Studio Audio Generation...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json={"model": model_name, "keep_alive": 0}
                ) as resp:
                    if resp.status == 200:
                        logger.debug("%s ejected from VRAM.", model_name)
            await asyncio.sleep(1)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.debug("Unload failed: %s", e)

    async def generate_melody(self, prompt: str) -> str:
        await self._unload_ollama()
        filename = f"melody_{int(time.time())}.wav"
        path = self.output_dir / filename
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(HEAVY_EXECUTOR, self._generate_melody_local, prompt, str(path))
            return result
        except Exception as e:
            logger.error("Melody generation failed: %s", e)
            return ""
        finally:
            self.clear_vram()

    async def generate_song_clip(
        self,
        vibe: str,
        bpm: int,
        voice_style: str,
        vocal_mode: str,
        lyrics: str = ""
    ) -> str:
        await self._unload_ollama()
        filename = f"kiba_studio_{int(time.time())}.mp3"
        path = self.output_dir / filename
        prompt = f"{vibe}, {voice_style} vocals, {vocal_mode}, {bpm} BPM. {lyrics}"
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(HEAVY_EXECUTOR, self._generate_yue_studio, prompt, str(path))
            return result
        except Exception as e:
            logger.error("YuE Studio generation failed: %s", e)
            return ""
        finally:
            self.clear_vram()

    def update_studio_settings(self, bpm: int = None, voice: str = None, mode: str = None):
        if bpm:
            self.bpm = max(60, min(200, bpm))
        if voice:
            self.voice_style = voice.strip().lower()
        if mode:
            self.vocal_mode = mode.strip().lower()
        logger.debug("Studio Config Updated: %s BPM | %s | %s", self.bpm, self.voice_style, self.vocal_mode)

    def _generate_melody_local(self, prompt: str, filepath: str) -> str:
        if StableAudioPipeline is None:
            logger.error("StableAudioPipeline not available. Check diffusers install.")
            return ""
        if self.active_model_type != "stable-audio":
            self.pipeline = StableAudioPipeline.from_pretrained(
                "stabilityai/stable-audio-open-1.0",
                torch_dtype=torch.float16
            )
            self.pipeline.to(self.device)
            self.active_model_type = "stable-audio"

        audio = self.pipeline(prompt, num_inference_steps=100, audio_end_in_s=15).audios[0]
        self._save_audio(audio, filepath)
        return filepath

    def _generate_yue_studio(self, prompt: str, filepath: str) -> str:
        """Calls YuE infer.py via subprocess — actual supported inference method."""
        if not os.path.exists(YUE_REPO_PATH):
            logger.error("YuE repo not found at %s — run the clone step first.", YUE_REPO_PATH)
            return ""

        # Split prompt into genre tags and lyrics on the period
        parts = prompt.split(".", 1)
        genre_text = parts[0].strip() if parts else "pop upbeat female vocal bright"
        lyrics_text = parts[1].strip() if len(parts) > 1 else prompt

        output_dir = str(self.output_dir / f"yue_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)

        genre_file = None
        lyrics_file = None

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as gf:
                gf.write(genre_text)
                genre_file = gf.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as lf:
                lf.write(lyrics_text)
                lyrics_file = lf.name

            venv_python = str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe")
            cmd = [
                venv_python, "infer.py",
                "--cuda_idx", "0",
                "--stage1_model", YUE_STAGE1_MODEL,
                "--stage2_model", YUE_STAGE2_MODEL,
                "--genre_txt", genre_file,
                "--lyrics_txt", lyrics_file,
                "--run_n_segments", "2",
                "--stage2_batch_size", "4",
                "--output_dir", output_dir,
                "--max_new_tokens", "3000",
                "--repetition_penalty", "1.1",
            ]

            logger.debug("Starting YuE inference — this takes ~6 minutes on 3090 Ti...")
            result = subprocess.run(
                cmd,
                cwd=YUE_REPO_PATH,
                capture_output=True,
                text=True,
                timeout=900
            )

            if result.returncode != 0:
                logger.error("YuE infer.py failed:\n%s", result.stderr)
                return ""

            # Find output file
            output_files = list(Path(output_dir).glob("**/*.mp3"))
            if not output_files:
                output_files = list(Path(output_dir).glob("**/*.wav"))
            if not output_files:
                logger.error("YuE ran but produced no output in %s", output_dir)
                return ""

            shutil.move(str(output_files[0]), filepath)
            return filepath

        except subprocess.TimeoutExpired:
            logger.error("YuE timed out after 15 minutes.")
            return ""
        except Exception as e:
            logger.error("YuE subprocess error: %s", e)
            return ""
        finally:
            if genre_file and os.path.exists(genre_file):
                os.unlink(genre_file)
            if lyrics_file and os.path.exists(lyrics_file):
                os.unlink(lyrics_file)

    def _save_audio(self, audio_data, filepath: str):
        import scipy.io.wavfile as wavfile
        wavfile.write(filepath, 44100, audio_data.cpu().numpy())

    def clear_vram(self):
        self.pipeline = None
        self.active_model_type = None
        torch.cuda.empty_cache()
        gc.collect()

    def _record_duration(self, name: str, started_at: float) -> None:
        if self.performance_tracker is None:
            return
        self.performance_tracker.record_service_call(
            name,
            (time.perf_counter() - started_at) * 1000,
        )