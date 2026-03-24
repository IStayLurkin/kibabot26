from __future__ import annotations

import logging
import time
from typing import Any

from core.config import MEDIA_SAFETY_MODE
from services.media_safety_service import format_media_error, is_moderation_error

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self, llm_service: Any | None = None, performance_tracker=None) -> None:
        self.llm_service = llm_service
        self.performance_tracker = performance_tracker

    async def generate_video(self, prompt: str) -> str:
        raise NotImplementedError("Video generation is not yet available. No backend is configured.")
    def _normalize_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            file_path = result.get("file_path")
            if isinstance(file_path, str) and file_path:
                return file_path

        raise RuntimeError("Unsupported video generation result format.")
