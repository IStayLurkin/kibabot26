from __future__ import annotations

import asyncio
import base64
import time

import httpx

from core.config import (
    OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_NUM_CTX, OLLAMA_REQUEST_TIMEOUT_SECONDS,
    VISION_FAST_MODEL, VISION_BEST_MODEL,
)
from core.logging_config import get_logger
from openai import OpenAI

logger = get_logger(__name__)

VISION_TIERS = {
    "fast": VISION_FAST_MODEL,
    "best": VISION_BEST_MODEL,
}


class VisionService:
    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS)

    async def _fetch_image_b64(self, url: str) -> tuple[str, str]:
        """Download image from URL and return (base64 string, content_type)."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/png").split(";")[0].strip()
            return base64.b64encode(resp.content).decode("utf-8"), content_type

    def _analyze_sync(self, image_b64: str, prompt: str, model: str, content_type: str = "image/png") -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "Describe this image."},
                        {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{image_b64}"}},
                    ],
                }
            ],
            max_tokens=1024,
            extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
        )
        return response.choices[0].message.content or ""

    async def analyze_url(self, url: str, prompt: str = "", tier: str = "fast") -> str:
        """Download image from URL and analyze it."""
        model = VISION_TIERS.get(tier, VISION_FAST_MODEL)
        started_at = time.perf_counter()
        try:
            image_b64, content_type = await self._fetch_image_b64(url)
            return await asyncio.to_thread(self._analyze_sync, image_b64, prompt, model, content_type)
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "vision.analyze_url",
                    (time.perf_counter() - started_at) * 1000,
                )

    async def analyze_bytes(self, image_bytes: bytes, prompt: str = "", content_type: str = "image/png", tier: str = "fast") -> str:
        """Analyze image from raw bytes."""
        model = VISION_TIERS.get(tier, VISION_FAST_MODEL)
        started_at = time.perf_counter()
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            return await asyncio.to_thread(self._analyze_sync, image_b64, prompt, model, content_type)
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "vision.analyze_bytes",
                    (time.perf_counter() - started_at) * 1000,
                )
