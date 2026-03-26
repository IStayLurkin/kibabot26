from __future__ import annotations

import asyncio
from pathlib import Path

from core.config import PARAKEET_MODEL
from core.logging_config import get_logger

logger = get_logger(__name__)

_nemo_model = None
_nemo_lock: asyncio.Lock | None = None


async def _get_nemo_model():
    global _nemo_model, _nemo_lock
    if _nemo_lock is None:
        _nemo_lock = asyncio.Lock()
    async with _nemo_lock:
        if _nemo_model is None:
            import nemo.collections.asr as nemo_asr
            _nemo_model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL)
            _nemo_model.eval()
            logger.info("[parakeet] Model loaded: %s", PARAKEET_MODEL)
    return _nemo_model


class ParakeetService:
    """STT via NVIDIA NeMo Parakeet — faster and more accurate than Whisper base."""

    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file. Returns transcript string."""
        try:
            model = await _get_nemo_model()
            def _run():
                result = model.transcribe([audio_path])
                # result is list of strings or list of Hypothesis objects
                if result and hasattr(result[0], "text"):
                    return result[0].text
                return str(result[0]) if result else ""
            transcript = await asyncio.to_thread(_run)
            logger.info("[parakeet] Transcribed %s → %r", audio_path, transcript[:80])
            return transcript
        except Exception as exc:
            logger.error("[parakeet] Transcription failed: %s", exc)
            return ""
