from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self, llm_service: Any | None = None, performance_tracker=None) -> None:
        self.llm_service = llm_service
        self.performance_tracker = performance_tracker

    async def generate_video(self, prompt: str) -> str:
        """
        Stub by design.

        Replace with your provider-specific async job flow later:
        - submit job
        - poll job
        - download video
        - return local file path
        """

        started_at = time.perf_counter()
        try:
            logger.info("Video generation requested: %s", prompt)
            raise NotImplementedError("Video generation is not wired yet.")
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "video.generate_video",
                    (time.perf_counter() - started_at) * 1000,
                )
