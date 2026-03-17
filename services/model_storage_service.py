from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

from core.config import ENABLED_MODEL_PROVIDERS, MODEL_PULL_TIMEOUT_SECONDS, MODEL_STORAGE_ROOT, OLLAMA_CLI_PATH
from core.logging_config import get_logger
from database.model_registry import get_model, upsert_model

logger = get_logger(__name__)


class ModelStorageService:
    def __init__(self, performance_tracker=None) -> None:
        self.performance_tracker = performance_tracker
        self.storage_root = Path(MODEL_STORAGE_ROOT)
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def initialize_storage(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        for provider in ENABLED_MODEL_PROVIDERS:
            (self.storage_root / provider).mkdir(parents=True, exist_ok=True)

    def provider_storage_dir(self, provider: str) -> Path:
        path = self.storage_root / provider
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def ensure_model_available(self, provider: str, model_name: str, model_type: str) -> tuple[bool, str]:
        model = await get_model(provider, model_name, model_type)
        if model is None:
            return False, f"Model `{provider}:{model_name}` is not registered."

        if self.is_model_available_locally(model):
            return True, f"Model `{provider}:{model_name}` is ready."

        return await self.pull_model(provider, model_name, model_type)

    def is_model_available_locally(self, model: dict) -> bool:
        provider = model["provider"]
        model_name = model["model_name"]
        local_path = model.get("local_path", "")

        if provider == "local":
            return bool(local_path and Path(local_path).exists())

        if provider == "ollama":
            return self._ollama_manifest_path(model_name).exists()

        if provider in {"hf", "automatic1111", "comfyui"}:
            return True

        return False

    async def pull_model(self, provider: str, model_name: str, model_type: str) -> tuple[bool, str]:
        provider = provider.strip().lower()
        if provider not in ENABLED_MODEL_PROVIDERS:
            return False, f"Provider `{provider}` is not enabled."

        if provider == "ollama":
            return await self._pull_ollama_model(model_name, model_type)

        if provider == "local":
            return False, (
                f"Local model `{model_name}` is not present in `{self.provider_storage_dir('local')}`. "
                "Place the model artifact there first."
            )

        if provider in {"hf", "automatic1111", "comfyui"}:
            return True, f"Provider `{provider}` does not require a local pull step for `{model_name}`."

        return False, f"Provider `{provider}` does not support pull/install yet."

    async def register_local_model(self, provider: str, model_name: str, model_type: str, local_path: str):
        await upsert_model(
            provider,
            model_name,
            model_type,
            source="disk",
            enabled=True,
            local_path=local_path,
            capabilities=["chat", "code"] if model_type == "llm" else ["image"],
            backend=provider,
        )

    async def _pull_ollama_model(self, model_name: str, model_type: str) -> tuple[bool, str]:
        manifest_path = self._ollama_manifest_path(model_name)
        started = asyncio.get_running_loop().time()
        logger.info("Pulling AI model: ollama:%s", model_name)
        ollama_executable = self._resolve_ollama_cli()
        if ollama_executable is None:
            return False, "The `ollama` CLI is not installed or not available on PATH."

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                [ollama_executable, "pull", model_name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=MODEL_PULL_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError:
            return False, "The `ollama` CLI is not installed or not available on PATH."
        except subprocess.TimeoutExpired:
            return False, f"Ollama pull timed out after {MODEL_PULL_TIMEOUT_SECONDS} seconds."
        except Exception as exc:
            return False, f"Ollama pull failed: {exc}"

        duration_ms = (asyncio.get_running_loop().time() - started) * 1000
        if self.performance_tracker is not None:
            self.performance_tracker.record_service_call("model_storage.pull.ollama", duration_ms)

        if completed.returncode != 0:
            stderr_text = (completed.stderr or completed.stdout or "").strip()
            return False, f"Ollama pull failed for `{model_name}`: {stderr_text or 'unknown error'}"

        manifest_path.write_text(
            json.dumps(
                {
                    "provider": "ollama",
                    "model_name": model_name,
                    "model_type": model_type,
                    "status": "ready",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Model ready: ollama:%s", model_name)
        return True, f"Pulled and prepared `ollama:{model_name}`."

    def _ollama_manifest_path(self, model_name: str) -> Path:
        safe_name = model_name.replace(":", "__")
        return self.provider_storage_dir("ollama") / f"{safe_name}.json"

    def _resolve_ollama_cli(self) -> str | None:
        if OLLAMA_CLI_PATH:
            configured = Path(OLLAMA_CLI_PATH)
            if configured.exists():
                return str(configured)

        on_path = shutil.which("ollama")
        if on_path:
            return on_path

        candidate_paths = [
            Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
        ]

        for candidate in candidate_paths:
            if candidate.exists():
                return str(candidate)

        return None
