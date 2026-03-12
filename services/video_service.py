from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self, llm_service: Any | None = None) -> None:
        self.llm_service = llm_service

    async def generate_video(self, prompt: str) -> str:
        """
        Stub by design.

        Replace with your provider-specific async job flow later:
        - submit job
        - poll job
        - download video
        - return local file path
        """
        logger.info("Video generation requested: %s", prompt)
        raise NotImplementedError("Video generation is not wired yet.")