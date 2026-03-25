from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VideoService:
    """Legacy video service placeholder.

    Real generation is handled by CogVideoService, AnimateDiffService, WanService.
    Use !cogvideo2b, !cogvideo5b, !animatediff, or !wan.
    """

    def __init__(self, llm_service: Any | None = None, performance_tracker=None) -> None:
        self.llm_service = llm_service
        self.performance_tracker = performance_tracker

    async def generate_video(self, prompt: str) -> str:
        raise NotImplementedError(
            "Use !cogvideo2b, !cogvideo5b, !animatediff, or !wan for video generation."
        )
