from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from urllib.parse import urlparse

from core.config import AUTOMATIC1111_BASE_URL, COMFYUI_BASE_URL, OLLAMA_BASE_URL
from core.logging_config import get_logger

logger = get_logger(__name__)


class HardwareService:
    def __init__(self) -> None:
        self._cached_status: dict | None = None

    async def get_status(self, refresh: bool = False) -> dict:
        if self._cached_status is not None and not refresh:
            return self._cached_status

        self._cached_status = await asyncio.to_thread(self._detect_status)
        return self._cached_status

    def _detect_status(self) -> dict:
        torch_status = self._detect_torch_cuda()
        nvidia_status = self._detect_nvidia_smi()
        ollama_status = self._detect_ollama_status()

        cuda_available = bool(torch_status["cuda_available"] or nvidia_status["gpu_visible"])
        gpu_name = torch_status["gpu_name"] or nvidia_status["gpu_name"]

        return {
            "cuda_available": cuda_available,
            "gpu_visible": bool(gpu_name),
            "gpu_name": gpu_name or "",
            "torch_available": torch_status["torch_available"],
            "torch_cuda_available": torch_status["cuda_available"],
            "torch_device_count": torch_status["device_count"],
            "nvidia_smi_available": nvidia_status["nvidia_smi_available"],
            "ollama_available": ollama_status["available"],
            "ollama_models": ollama_status["models"],
            "ollama_error": ollama_status["error"],
            "automatic1111_available": self._detect_simple_json_endpoint(
                AUTOMATIC1111_BASE_URL,
                "/sdapi/v1/sd-models",
            ),
            "comfyui_available": self._detect_simple_json_endpoint(
                COMFYUI_BASE_URL,
                "/object_info",
            ),
        }

    def _detect_torch_cuda(self) -> dict:
        if importlib.util.find_spec("torch") is None:
            return {
                "torch_available": False,
                "cuda_available": False,
                "device_count": 0,
                "gpu_name": "",
            }

        try:
            import torch
        except Exception as exc:
            logger.debug("Torch import failed during CUDA detection: %s", exc)
            return {
                "torch_available": False,
                "cuda_available": False,
                "device_count": 0,
                "gpu_name": "",
            }

        try:
            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count()) if cuda_available else 0
            gpu_name = torch.cuda.get_device_name(0) if cuda_available and device_count else ""
            return {
                "torch_available": True,
                "cuda_available": cuda_available,
                "device_count": device_count,
                "gpu_name": gpu_name,
            }
        except Exception as exc:
            logger.debug("Torch CUDA detection failed: %s", exc)
            return {
                "torch_available": True,
                "cuda_available": False,
                "device_count": 0,
                "gpu_name": "",
            }

    def _detect_nvidia_smi(self) -> dict:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return {
                "nvidia_smi_available": False,
                "gpu_visible": False,
                "gpu_name": "",
            }

        try:
            result = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception as exc:
            logger.debug("nvidia-smi detection failed: %s", exc)
            return {
                "nvidia_smi_available": True,
                "gpu_visible": False,
                "gpu_name": "",
            }

        if result.returncode != 0:
            return {
                "nvidia_smi_available": True,
                "gpu_visible": False,
                "gpu_name": "",
            }

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {
            "nvidia_smi_available": True,
            "gpu_visible": bool(lines),
            "gpu_name": lines[0] if lines else "",
        }

    def _detect_ollama_status(self) -> dict:
        parsed = urlparse(OLLAMA_BASE_URL)
        base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "http://localhost:11434"
        tags_url = f"{base}/api/tags"
        request = urllib.request.Request(tags_url, headers={"User-Agent": "KibaBot/1.0"})

        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            return {
                "available": False,
                "models": [],
                "error": str(exc),
            }

        models = payload.get("models", [])
        names = []
        for model in models:
            if isinstance(model, dict) and model.get("name"):
                names.append(str(model["name"]))

        return {
            "available": True,
            "models": names,
            "error": "",
        }

    def get_vram_usage_mb(self) -> int:
        """Returns current GPU VRAM usage in MB via nvidia-smi, or 0 on failure."""
        executable = shutil.which("nvidia-smi")
        if not executable:
            return 0
        try:
            result = subprocess.run(
                [executable, "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as exc:
            logger.debug("VRAM usage query failed: %s", exc)
        return 0

    def get_ollama_running_models(self) -> list[dict]:
        """Returns models currently loaded in Ollama VRAM via /api/ps. Returns [] on failure."""
        parsed = urlparse(OLLAMA_BASE_URL)
        base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "http://localhost:11434"
        url = f"{base}/api/ps"
        request = urllib.request.Request(url, headers={"User-Agent": "KibaBot/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
                return payload.get("models", [])
        except Exception as exc:
            logger.debug("Ollama /api/ps failed: %s", exc)
            return []

    def _detect_simple_json_endpoint(self, base_url: str, path: str) -> bool:
        base_url = base_url.strip()
        if not base_url:
            return False

        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            return False

        url = f"{parsed.scheme}://{parsed.netloc}{path}"
        request = urllib.request.Request(url, headers={"User-Agent": "KibaBot/1.0"})

        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                if response.status < 200 or response.status >= 300:
                    return False
                json.loads(response.read().decode("utf-8", errors="replace"))
                return True
        except Exception:
            return False
