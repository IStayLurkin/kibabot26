from __future__ import annotations

import logging
import os
import time
import uuid
import wave
from math import pi, sin
from pathlib import Path
from typing import Any

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

    async def text_to_speech(self, text: str) -> str:
        """
        Adapter-first design.

        Expected provider return formats:

        1. local file path string
        2. raw bytes
        3. {"file_path": "..."}
        4. {"audio_bytes": b"..."}
        """

        started_at = time.perf_counter()
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
        print(f"[DEBUG] Voice/TTS Request -> Engine: {self.provider} | Local: True")
        print(f"[DEBUG] Voice/TTS Request -> Engine: {self.provider} | Audioop: Native (3.12)")
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
