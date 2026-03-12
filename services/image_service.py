from __future__ import annotations

import base64
import logging
import os
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from core.feature_flags import MEDIA_OUTPUT_DIR

logger = logging.getLogger(__name__)


class ImageService:
    def __init__(self, llm_service: Any | None = None, performance_tracker=None) -> None:
        self.llm_service = llm_service
        self.performance_tracker = performance_tracker
        self.output_dir = Path(MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_image(self, prompt: str) -> str:
        """
        Pluggable adapter layer.

        Expected provider returns:

        1. {"b64_json": "..."}
        2. {"image_base64": "..."}
        3. {"file_path": "..."}
        4. direct file path string
        """

        started_at = time.perf_counter()
        try:
            if self.llm_service is None:
                raise RuntimeError(
                    "Image service is not configured with an image-capable provider."
                )

            last_error: Exception | None = None
            for method_name in ("generate_image", "image", "create_image"):
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
                        logger.exception("Image generation failed via %s", method_name)

                except Exception as exc:
                    last_error = exc
                    logger.exception("Image generation failed via %s", method_name)

            if last_error is not None:
                raise RuntimeError(f"Image generation failed: {last_error}") from last_error

            raise RuntimeError("Image generation is not available on the current llm_service.")
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "image.generate_image",
                    (time.perf_counter() - started_at) * 1000,
                )

    def _normalize_result(self, result: Any) -> str:
        if isinstance(result, str):
            if os.path.exists(result):
                return result

            raise RuntimeError(
                f"Image provider returned string, but file does not exist: {result}"
            )

        if isinstance(result, dict):
            file_path = result.get("file_path")

            if isinstance(file_path, str) and os.path.exists(file_path):
                return file_path

            b64_data = result.get("b64_json") or result.get("image_base64")

            if isinstance(b64_data, str):
                return self._write_base64_png(b64_data)

            image_url = result.get("url")
            if isinstance(image_url, str) and image_url.strip():
                return self._download_image(image_url)

        raise RuntimeError("Unsupported image generation result format.")

    def _write_base64_png(self, b64_data: str) -> str:
        filename = f"image_{uuid.uuid4().hex}.png"
        path = self.output_dir / filename

        try:
            image_bytes = base64.b64decode(b64_data)
        except Exception as exc:
            raise RuntimeError("Failed to decode base64 image data.") from exc

        path.write_bytes(image_bytes)
        return str(path)

    def _download_image(self, image_url: str) -> str:
        filename = f"image_{uuid.uuid4().hex}.png"
        path = self.output_dir / filename

        request = urllib.request.Request(
            image_url,
            headers={"User-Agent": "KibaBot/1.0"},
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                image_bytes = response.read()
        except Exception as exc:
            raise RuntimeError(f"Failed to download generated image: {exc}") from exc

        path.write_bytes(image_bytes)
        return str(path)
