from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import httpx

from core.config import FISH_SPEECH_BASE_URL, MEDIA_OUTPUT_DIR
from core.logging_config import get_logger

logger = get_logger(__name__)


class FishSpeechService:
    """TTS via Fish Speech V1.5 local inference server."""

    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
        self.output_dir = Path(MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = FISH_SPEECH_BASE_URL

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/v1/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def synthesize(self, text: str, voice_id: str = "default") -> str | None:
        """Synthesize text to speech. Returns path to WAV file or None on failure."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/tts",
                    json={"text": text, "voice": voice_id, "format": "wav"},
                )
                resp.raise_for_status()
                out_path = self.output_dir / f"fish_{uuid.uuid4().hex[:8]}.wav"
                out_path.write_bytes(resp.content)
                logger.info("[fish_speech] Synthesized %d chars → %s", len(text), out_path)
                return str(out_path)
        except Exception as exc:
            logger.error("[fish_speech] Synthesis failed: %s", exc)
            return None
