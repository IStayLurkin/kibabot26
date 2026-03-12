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
        started_at = time.perf_counter()
        try:
            logger.info("Video generation requested: %s", prompt)
            if self.llm_service is None:
                raise RuntimeError("Video service is not configured with a video-capable provider.")

            last_error: Exception | None = None
            for method_name in ("generate_video", "video", "create_video"):
                method = getattr(self.llm_service, method_name, None)
                if method is None:
                    continue

                try:
                    result = await method(prompt=prompt)
                    return self._normalize_result(result)
                except TypeError as exc:
                    last_error = exc
                    try:
                        result = await method(prompt)
                        return self._normalize_result(result)
                    except Exception as inner_exc:
                        last_error = inner_exc
                        logger.exception("Video generation failed via %s", method_name)
                except Exception as exc:
                    last_error = exc
                    logger.exception("Video generation failed via %s", method_name)

            if last_error is not None:
                raise RuntimeError(f"Video generation failed: {last_error}") from last_error

            raise RuntimeError("Video generation is not available on the current llm_service.")
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "video.generate_video",
                    (time.perf_counter() - started_at) * 1000,
                )

    def _normalize_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            file_path = result.get("file_path")
            if isinstance(file_path, str) and file_path:
                return file_path

        raise RuntimeError("Unsupported video generation result format.")
