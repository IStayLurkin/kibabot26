from __future__ import annotations

import gc
import logging
import os
import time
import uuid
import wave
import asyncio
import subprocess
from math import pi, sin
from pathlib import Path
from typing import Any
from faster_whisper import WhisperModel  # pip install faster-whisper

try:
    import torch
except ImportError:
    torch = None

from core.config import MEDIA_SAFETY_MODE
from core.feature_flags import MEDIA_OUTPUT_DIR
from services.media_safety_service import format_media_error, is_moderation_error

logger = logging.getLogger(__name__)


class VoiceService:
    def __init__(self, llm_service: Any | None = None, performance_tracker=None) -> None:
        self.llm_service = llm_service
        self.performance_tracker = performance_tracker
        self.output_dir = Path(MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 2026 Hardware Optimization: Using 3090 Ti for Whisper inference
        self.stt_model = None  # Lazy-loaded on first use
        self._stt_last_used: float = 0.0
        self._stt_unload_task = None
        self._STT_IDLE_SECONDS = 300  # Unload Whisper after 5 min idle

    # --- NEW: 3090 Ti SPEECH TO TEXT ---
    async def speech_to_text(self, audio_path: str) -> str:
        """
        2026 Expansion: Converts user voice to text using local GPU.
        Decoupled from main loop to prevent stream lag.
        """
        if self.stt_model is None:
            self.stt_model = WhisperModel("base", device="cuda", compute_type="float16")
        started_at = time.perf_counter()
        loop = asyncio.get_running_loop()
        def transcribe():
            segments, _info = self.stt_model.transcribe(audio_path, beam_size=5)
            return " ".join([s.text for s in segments])

        from core.executors import HEAVY_EXECUTOR
        result = await asyncio.wait_for(
            loop.run_in_executor(HEAVY_EXECUTOR, transcribe),
            timeout=120.0,
        )
        self._stt_last_used = time.time()
        if self._stt_unload_task:
            self._stt_unload_task.cancel()
        task = asyncio.create_task(self._stt_inactivity_monitor())
        task.add_done_callback(self._on_stt_monitor_done)
        self._stt_unload_task = task
        self._record_duration("voice.speech_to_text", started_at)
        return result

    async def text_to_speech(self, text: str) -> str:
        """
        Adapter-first design upgraded with 2026 Local Piper support.
        Preserves original fallback loop.
        """

        started_at = time.perf_counter()
        
        # --- NEW: TRY LOCAL PIPER FIRST ---
        proc = None
        try:
            filename = f"piper_{uuid.uuid4().hex}.wav"
            path = self.output_dir / filename
            command = ["piper", "--model", "en_US-kiba-medium.onnx", "--output_file", str(path)]
            proc = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.PIPE)
            await asyncio.wait_for(proc.communicate(input=text.encode()), timeout=30.0)
            if path.exists():
                return str(path)
        except asyncio.TimeoutError:
            logger.warning("Local Piper TTS timed out after 30s, falling back")
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Local Piper TTS failed, falling back to original logic: %s", e)
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass

        # --- PRESERVED ORIGINAL PROVIDER LOOP ---
        try:
            last_error: Exception | None = None
            if self.llm_service is not None:
                for method_name in ("text_to_speech", "generate_speech", "tts"):
                    method = getattr(self.llm_service, method_name, None)

                    if method is None:
                        continue

                    try:
                        result = await method(text=text)
                        return self._normalize_result(result)

                    except TypeError as exc:
                        last_error = exc
                        try:
                            result = await method(text)
                            return self._normalize_result(result)
                        except Exception as inner_exc:
                            last_error = inner_exc
                            if is_moderation_error(inner_exc):
                                logger.warning("TTS prompt blocked via %s", method_name)
                            else:
                                logger.exception("TTS failed via %s", method_name)

                    except Exception as exc:
                        last_error = exc
                        if is_moderation_error(exc):
                            logger.warning("TTS prompt blocked via %s", method_name)
                        else:
                            logger.exception("TTS failed via %s", method_name)

                if last_error is not None:
                    raise RuntimeError(format_media_error(last_error, "audio", MEDIA_SAFETY_MODE)) from last_error

            logger.warning(
                "No TTS provider configured. Returning placeholder audio tone."
            )

            return self._build_placeholder_wav()
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "voice.text_to_speech",
                    (time.perf_counter() - started_at) * 1000,
                )

    def _normalize_result(self, result: Any) -> str:
        if isinstance(result, str):
            if os.path.exists(result):
                return result

            raise RuntimeError(
                f"TTS provider returned string, but file does not exist: {result}"
            )

        if isinstance(result, bytes):
            return self._write_audio_bytes(result)

        if isinstance(result, dict):
            file_path = result.get("file_path")

            if isinstance(file_path, str) and os.path.exists(file_path):
                return file_path

            audio_bytes = result.get("audio_bytes")

            if isinstance(audio_bytes, bytes):
                return self._write_audio_bytes(audio_bytes)

        raise RuntimeError("Unsupported TTS result format.")

    def _write_audio_bytes(self, data: bytes, extension: str = ".mp3") -> str:
        filename = f"tts_{uuid.uuid4().hex}{extension}"
        path = self.output_dir / filename

        path.write_bytes(data)

        return str(path)

    def _build_placeholder_wav(self) -> str:
        filename = f"tts_placeholder_{uuid.uuid4().hex}.wav"
        path = self.output_dir / filename

        framerate = 44100
        duration_seconds = 1.2
        frequency = 440.0
        amplitude = 12000

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(framerate)

            frames = bytearray()
            total_frames = int(duration_seconds * framerate)

            for i in range(total_frames):
                sample = int(
                    amplitude * sin(2 * pi * frequency * (i / framerate))
                )

                frames.extend(
                    sample.to_bytes(2, byteorder="little", signed=True)
                )

            wav_file.writeframes(bytes(frames))

        return str(path)

    async def _stt_inactivity_monitor(self):
        """Unloads the Whisper model after idle for _STT_IDLE_SECONDS."""
        await asyncio.sleep(self._STT_IDLE_SECONDS)
        if self.stt_model is not None:
            self.stt_model = None
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("[VoiceService] Whisper unloaded after %ss idle.", self._STT_IDLE_SECONDS)

    @staticmethod
    def _on_stt_monitor_done(t: asyncio.Task):
        if not t.cancelled() and t.exception() is not None:
            logger.exception("[VoiceService] STT inactivity monitor raised", exc_info=t.exception())

    def _record_duration(self, name: str, started_at: float) -> None:
        if self.performance_tracker is None:
            return

        self.performance_tracker.record_service_call(
            name,
            (time.perf_counter() - started_at) * 1000,
        )