from __future__ import annotations

import asyncio
import re
import time
from core.config import (
    OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_NUM_CTX, OLLAMA_REQUEST_TIMEOUT_SECONDS,
    THINKING_FAST_MODEL, THINKING_BEST_MODEL,
)
from core.logging_config import get_logger
from openai import OpenAI

logger = get_logger(__name__)

THINKING_TIERS = {
    "fast": THINKING_FAST_MODEL,
    "best": THINKING_BEST_MODEL,
}


class ThinkingService:
    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS)

    def _think_sync(self, prompt: str, tier: str) -> str:
        model = THINKING_TIERS.get(tier, THINKING_FAST_MODEL)
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=4096,
            extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX * 2}},
        )
        return response.choices[0].message.content or ""

    async def think(self, prompt: str, tier: str = "fast") -> str:
        """Run a thinking/reasoning model. Returns final answer with <think> blocks stripped."""
        started_at = time.perf_counter()
        try:
            raw = await asyncio.to_thread(self._think_sync, prompt, tier)
            # Strip <think>...</think> blocks — already handled by _sanitize_model_text but do it here too
            cleaned = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
            return cleaned if cleaned else raw.strip()
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "thinking.think",
                    (time.perf_counter() - started_at) * 1000,
                )
