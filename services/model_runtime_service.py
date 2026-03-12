from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.config import (
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_DEFAULT_MODEL,
    COMFYUI_BASE_URL,
    COMFYUI_DEFAULT_MODEL,
    CUDA_PREFERRED,
    HF_MODEL,
    IMAGE_PROVIDER,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OPENAI_IMAGE_MODEL,
    OPENAI_MODEL,
    PREFERRED_LOCAL_IMAGE_BACKEND,
)
from core.logging_config import get_logger
from database.model_registry import (
    find_models,
    get_model,
    get_runtime_settings,
    list_models,
    set_runtime_setting,
    upsert_model,
)
from services.hardware_service import HardwareService

logger = get_logger(__name__)

LOCAL_MODEL_STORAGE_DIR = Path("bot_dl_storage")
LLM_DIR = LOCAL_MODEL_STORAGE_DIR / "llm"
IMAGE_DIR = LOCAL_MODEL_STORAGE_DIR / "image"
IMAGE_HINTS = ("sd", "sdxl", "flux", "stable-diffusion")
LOCAL_IMAGE_PROVIDERS = {"local", "ollama", "automatic1111", "comfyui"}
REMOTE_PROVIDERS = {"openai", "hf"}


@dataclass(slots=True)
class RuntimeModelState:
    active_llm_provider: str
    active_llm_model: str
    active_image_provider: str
    active_image_model: str
    preferred_compute_device: str
    cuda_preferred: bool
    cuda_enabled: bool
    gpu_name: str
    ollama_available: bool
    active_device: str
    provider_status: dict[str, bool] = field(default_factory=dict)


class ModelRuntimeService:
    def __init__(self, hardware_service: HardwareService, performance_tracker=None) -> None:
        self.hardware_service = hardware_service
        self.performance_tracker = performance_tracker
        self.state = RuntimeModelState(
            active_llm_provider=LLM_PROVIDER,
            active_llm_model=self._default_llm_model_for_provider(LLM_PROVIDER),
            active_image_provider=IMAGE_PROVIDER,
            active_image_model=OPENAI_IMAGE_MODEL,
            preferred_compute_device="cuda" if CUDA_PREFERRED else "cpu",
            cuda_preferred=CUDA_PREFERRED,
            cuda_enabled=False,
            gpu_name="",
            ollama_available=False,
            active_device="Remote",
            provider_status={},
        )
        self.last_openai_usage: dict[str, str | int] = {}
        self.last_openai_rate_limits: dict[str, str] = {}

    async def initialize(self):
        await self.register_default_models()
        await self.refresh_hardware_status()
        await self.scan_local_storage()
        await self.sync_models("llm")
        await self.sync_models("image")
        await self.load_runtime_settings()

    async def register_default_models(self):
        defaults = [
            ("openai", OPENAI_MODEL, "llm", "default", None, ["chat", "code"], "openai", ""),
            ("openai", OPENAI_IMAGE_MODEL, "image", "default", None, ["image"], "openai", ""),
            ("ollama", OLLAMA_MODEL, "llm", "default", None, ["chat", "code"], "ollama", "cuda"),
            ("hf", HF_MODEL, "llm", "default", None, ["chat", "code"], "hf", ""),
        ]

        if AUTOMATIC1111_BASE_URL:
            defaults.append(
                ("automatic1111", AUTOMATIC1111_DEFAULT_MODEL, "image", "default", None, ["image"], "automatic1111", "cuda")
            )

        if COMFYUI_BASE_URL:
            defaults.append(
                ("comfyui", COMFYUI_DEFAULT_MODEL, "image", "default", None, ["image"], "comfyui", "cuda")
            )

        for provider, model_name, model_type, source, local_path, capabilities, backend, preferred_device in defaults:
            await upsert_model(
                provider,
                model_name,
                model_type,
                source=source,
                enabled=True,
                local_path=local_path,
                capabilities=capabilities,
                backend=backend,
                preferred_device=preferred_device,
            )
            logger.debug("[model_registered] provider=%s model=%s type=%s source=%s", provider, model_name, model_type, source)

    async def load_runtime_settings(self):
        settings = await get_runtime_settings()
        self.state.active_llm_provider = settings.get("active_llm_provider", self.state.active_llm_provider)
        self.state.active_llm_model = settings.get("active_llm_model", self.state.active_llm_model)
        self.state.active_image_provider = settings.get("active_image_provider", self.state.active_image_provider)
        self.state.active_image_model = settings.get("active_image_model", self.state.active_image_model)
        self.state.preferred_compute_device = settings.get("preferred_compute_device", self.state.preferred_compute_device)
        self.state.cuda_preferred = settings.get("cuda_preferred", str(self.state.cuda_preferred)).lower() in {"1", "true", "yes", "on"}
        self._apply_device_selection()

    async def persist_state(self):
        await set_runtime_setting("active_llm_provider", self.state.active_llm_provider)
        await set_runtime_setting("active_llm_model", self.state.active_llm_model)
        await set_runtime_setting("active_image_provider", self.state.active_image_provider)
        await set_runtime_setting("active_image_model", self.state.active_image_model)
        await set_runtime_setting("preferred_compute_device", self.state.preferred_compute_device)
        await set_runtime_setting("cuda_preferred", "true" if self.state.cuda_preferred else "false")

    async def refresh_hardware_status(self):
        hardware = await self.hardware_service.get_status(refresh=True)
        self.state.cuda_enabled = bool(hardware["cuda_available"])
        self.state.gpu_name = hardware["gpu_name"]
        self.state.ollama_available = bool(hardware["ollama_available"])
        self.state.provider_status = {
            "ollama": bool(hardware["ollama_available"]),
            "automatic1111": bool(hardware.get("automatic1111_available")),
            "comfyui": bool(hardware.get("comfyui_available")),
            "openai": True,
            "hf": True,
            "local": self.state.cuda_enabled,
        }
        logger.info("[cuda_detected] enabled=%s gpu=%s", "yes" if self.state.cuda_enabled else "no", self.state.gpu_name or "unknown")
        self._apply_device_selection()
        return hardware

    def _apply_device_selection(self):
        active_providers = {self.state.active_llm_provider, self.state.active_image_provider}
        local_backend_active = any(provider in LOCAL_IMAGE_PROVIDERS for provider in active_providers)

        if not local_backend_active:
            self.state.active_device = "Remote"
            return

        preferred = "cuda" if self.state.cuda_preferred else self.state.preferred_compute_device
        if preferred == "cuda" and self.state.cuda_enabled:
            self.state.preferred_compute_device = "cuda"
            self.state.active_device = "CUDA"
        else:
            self.state.preferred_compute_device = "cpu"
            self.state.active_device = "CPU"

        logger.info(
            "[device_selected] preferred=%s active=%s gpu=%s",
            self.state.preferred_compute_device.upper(),
            self.state.active_device,
            self.state.gpu_name or "none",
        )

    async def scan_local_storage(self):
        for directory in (LLM_DIR, IMAGE_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        await self._scan_llm_directory()
        await self._scan_image_directory()

    async def _scan_llm_directory(self):
        valid_suffixes = {".gguf", ".bin", ".safetensors", ".pt", ".pth"}
        for path in sorted(LLM_DIR.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in valid_suffixes:
                model_name = path.stem
            elif path.is_dir():
                model_name = path.name
            else:
                continue

            await upsert_model(
                "local",
                model_name,
                "llm",
                source="disk",
                enabled=True,
                local_path=str(path),
                capabilities=["chat", "code"],
                backend="local",
                preferred_device=self.state.preferred_compute_device,
            )
            logger.info("[model_discovered] provider=local type=llm model=%s", model_name)
            logger.info("[local_model_detected] provider=local type=llm model=%s path=%s", model_name, path)

    async def _scan_image_directory(self):
        for path in sorted(IMAGE_DIR.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_dir() and path.suffix.lower() not in {".safetensors", ".ckpt", ".pt"}:
                continue

            model_name = path.stem if path.is_file() else path.name
            await upsert_model(
                "local",
                model_name,
                "image",
                source="disk",
                enabled=True,
                local_path=str(path),
                capabilities=["image"],
                backend=PREFERRED_LOCAL_IMAGE_BACKEND,
                preferred_device=self.state.preferred_compute_device,
            )
            logger.info("[model_discovered] provider=local type=image model=%s", model_name)
            logger.info("[local_model_detected] provider=local type=image model=%s path=%s", model_name, path)

    async def sync_models(self, model_type: str) -> dict:
        discovered = []

        if model_type == "llm":
            hardware = await self.hardware_service.get_status(refresh=True)
            if hardware["ollama_available"]:
                for model_name in hardware["ollama_models"]:
                    await upsert_model(
                        "ollama",
                        model_name,
                        "llm",
                        source="discovered",
                        enabled=True,
                        capabilities=["chat", "code"],
                        backend="ollama",
                        preferred_device="cuda" if self.state.cuda_enabled else "cpu",
                        update_last_synced=True,
                    )
                    logger.info("[model_discovered] provider=ollama type=llm model=%s", model_name)
                    logger.info("[ollama_model_discovered] provider=ollama model=%s", model_name)
                    discovered.append(f"ollama:{model_name}")
            await self._scan_llm_directory()

        if model_type == "image":
            await self._scan_image_directory()
            for model in await list_models("llm"):
                lowered = model["model_name"].lower()
                if model["provider"] == "ollama" and any(hint in lowered for hint in IMAGE_HINTS):
                    await upsert_model(
                        "ollama",
                        model["model_name"],
                        "image",
                        source="discovered",
                        enabled=True,
                        capabilities=["image"],
                        backend="ollama",
                        preferred_device="cuda" if self.state.cuda_enabled else "cpu",
                        update_last_synced=True,
                    )
                    logger.info("[model_discovered] provider=ollama type=image model=%s", model["model_name"])
                    logger.info("[ollama_model_discovered] provider=ollama model=%s", model["model_name"])
                    discovered.append(f"ollama:{model['model_name']}")

            await upsert_model(
                "openai",
                OPENAI_IMAGE_MODEL,
                "image",
                source="default",
                enabled=True,
                capabilities=["image"],
                backend="openai",
                preferred_device="",
                update_last_synced=True,
            )

            if AUTOMATIC1111_BASE_URL:
                await upsert_model(
                    "automatic1111",
                    AUTOMATIC1111_DEFAULT_MODEL,
                    "image",
                    source="default",
                    enabled=True,
                    capabilities=["image"],
                    backend="automatic1111",
                    preferred_device="cuda" if self.state.cuda_enabled else "cpu",
                    update_last_synced=True,
                )
                discovered.append(f"automatic1111:{AUTOMATIC1111_DEFAULT_MODEL}")

            if COMFYUI_BASE_URL:
                await upsert_model(
                    "comfyui",
                    COMFYUI_DEFAULT_MODEL,
                    "image",
                    source="default",
                    enabled=True,
                    capabilities=["image"],
                    backend="comfyui",
                    preferred_device="cuda" if self.state.cuda_enabled else "cpu",
                    update_last_synced=True,
                )
                discovered.append(f"comfyui:{COMFYUI_DEFAULT_MODEL}")

        return {"model_type": model_type, "discovered": discovered, "count": len(discovered)}

    async def add_model(self, provider: str, model_name: str, model_type: str):
        provider = provider.strip().lower()
        model_name = model_name.strip()
        preferred_device = ""
        if provider in {"local", "ollama", "automatic1111", "comfyui"}:
            preferred_device = "cuda" if self.state.cuda_enabled else "cpu"

        await upsert_model(
            provider,
            model_name,
            model_type,
            source="manual",
            enabled=True,
            capabilities=["chat", "code"] if model_type == "llm" else ["image"],
            backend=provider,
            preferred_device=preferred_device,
        )
        logger.info("[model_registered] provider=%s model=%s type=%s source=manual", provider, model_name, model_type)

    async def resolve_model(self, model_type: str, model_reference: str):
        cleaned = model_reference.strip()
        if ":" in cleaned:
            provider, model_name = cleaned.split(":", 1)
            provider = provider.strip().lower()
            model_name = model_name.strip()
            model = await get_model(provider, model_name, model_type)
            if model is None:
                matches = await list_models(model_type)
                provider_matches = [
                    item for item in matches
                    if item["provider"] == provider and item["model_name"].lower().startswith(model_name.lower())
                ]
                if len(provider_matches) == 1:
                    return provider_matches[0], ""
            if model is None:
                return None, f"I couldn't find `{cleaned}` in the {model_type} registry."
            return model, ""

        matches = await find_models(model_type, cleaned)
        enabled_matches = [model for model in matches if model["enabled"]]
        if not enabled_matches:
            all_models = await list_models(model_type)
            enabled_matches = [
                model for model in all_models
                if model["enabled"] and model["model_name"].lower().startswith(cleaned.lower())
            ]
        if not enabled_matches:
            return None, f"I couldn't find `{cleaned}` in the {model_type} registry."

        if len(enabled_matches) > 1:
            options = ", ".join(f"{model['provider']}:{model['model_name']}" for model in enabled_matches[:5])
            return None, f"`{cleaned}` matches multiple models. Use one of: {options}"

        return enabled_matches[0], ""

    async def set_active_model(self, model_type: str, model_reference: str):
        model, error = await self.resolve_model(model_type, model_reference)
        if model is None:
            return False, error

        availability = self._provider_ready(model["provider"])
        if model["provider"] in {"ollama", "automatic1111", "comfyui"} and not availability:
            return False, f"`{model['provider']}` is registered but not reachable right now."

        if model["provider"] == "local" and model_type == "image" and not (
            self._provider_ready("automatic1111") or self._provider_ready("comfyui")
        ):
            return False, "Local image models are registered, but no local image backend API is reachable right now."

        if model_type == "llm":
            self.state.active_llm_provider = model["provider"]
            self.state.active_llm_model = model["model_name"]
        else:
            self.state.active_image_provider = model["provider"]
            self.state.active_image_model = model["model_name"]

        if model["preferred_device"]:
            self.state.preferred_compute_device = model["preferred_device"]

        self._apply_device_selection()
        await self.persist_state()
        logger.info("[model_swap] type=%s provider=%s model=%s device=%s", model_type, model["provider"], model["model_name"], self.state.active_device)
        return True, f"Active {model_type} model set to {model['provider']}:{model['model_name']}."

    async def get_models(self, model_type: str):
        return await list_models(model_type)

    def get_current_model_text(self, model_type: str) -> str:
        if model_type == "llm":
            lines = [
                f"Active LLM provider: {self.state.active_llm_provider}",
                f"Model: {self.state.active_llm_model}",
            ]
            usage_text = self.get_openai_usage_text()
            if self.state.active_llm_provider == "openai" and usage_text:
                lines.append("")
                lines.append(usage_text)
            return "\n".join(lines)

        lines = [
            f"Active image provider: {self.state.active_image_provider}",
            f"Model: {self.state.active_image_model}",
        ]
        rate_text = self.get_openai_rate_limit_text()
        if self.state.active_image_provider == "openai" and rate_text:
            lines.append("")
            lines.append(rate_text)
        return "\n".join(lines)

    async def get_model_list_text(self, model_type: str) -> str:
        models = await self.get_models(model_type)
        if not models:
            return f"No {model_type} models are registered yet."

        active_provider = self.state.active_llm_provider if model_type == "llm" else self.state.active_image_provider
        active_model = self.state.active_llm_model if model_type == "llm" else self.state.active_image_model
        lines = [f"Available {model_type.upper()} models:"]
        for model in models:
            marker = " (active)" if model["provider"] == active_provider and model["model_name"] == active_model else ""
            source = f" [{model['source']}]" if model["source"] else ""
            lines.append(f"- {model['provider']}:{model['model_name']}{marker}{source}")
        return "\n".join(lines)

    async def get_hardware_status_text(self) -> str:
        hardware = await self.hardware_service.get_status(refresh=True)
        self.state.cuda_enabled = bool(hardware["cuda_available"])
        self.state.gpu_name = hardware["gpu_name"]
        self.state.ollama_available = bool(hardware["ollama_available"])
        self.state.provider_status = {
            "ollama": bool(hardware["ollama_available"]),
            "automatic1111": bool(hardware.get("automatic1111_available")),
            "comfyui": bool(hardware.get("comfyui_available")),
            "openai": True,
            "hf": True,
            "local": self.state.cuda_enabled,
        }
        self._apply_device_selection()
        local_backend_active = self._uses_local_backend()

        lines = [
            f"CUDA status: {'enabled' if self.state.cuda_enabled else 'disabled'}",
            f"Preferred device: {self.state.preferred_compute_device.upper()}",
            f"Active device: {self.state.active_device}",
            f"GPU detected: {self.state.gpu_name or 'Not detected'}",
            f"Ollama available: {'yes' if self.state.ollama_available else 'no'}",
            f"Automatic1111 available: {'yes' if self._provider_ready('automatic1111') else 'no'}",
            f"ComfyUI available: {'yes' if self._provider_ready('comfyui') else 'no'}",
            f"Local backend using GPU: {'yes' if local_backend_active and self.state.active_device == 'CUDA' else 'no'}",
        ]

        if self.state.active_device == "CPU" and self.state.cuda_enabled and local_backend_active:
            lines.append("Fallback: CPU only because the active backend or model does not currently support GPU execution.")
        elif self.state.active_device == "Remote":
            lines.append("Fallback: remote provider is active, so your local GPU is available but not currently used for inference.")
        elif self.state.active_device == "CPU":
            lines.append("Fallback: CPU only if needed.")

        return "\n".join(lines)

    def answer_natural_language_query(self, text: str) -> str | None:
        lowered = text.strip().lower()
        if lowered in {"what model are you using", "what llm are you using", "what model you using"}:
            return self.get_current_model_text("llm")
        if lowered in {"what image model are you using", "what image generator are you using"}:
            return self.get_current_model_text("image")
        if lowered in {"are you using ollama", "are you on ollama"}:
            return f"Active LLM provider: {self.state.active_llm_provider}\nOllama available: {'yes' if self.state.ollama_available else 'no'}"
        if "3090 ti" in lowered:
            return f"GPU detected: {self.state.gpu_name or 'Not detected'}\nActive device: {self.state.active_device}"
        if "is ollama available" in lowered:
            return f"Ollama available: {'yes' if self.state.ollama_available else 'no'}"
        if "token usage" in lowered or "rate limit" in lowered:
            details = [self.get_openai_usage_text(), self.get_openai_rate_limit_text()]
            rendered = "\n".join(part for part in details if part)
            return rendered or "I don't have recent OpenAI usage or rate-limit data yet."
        return None

    def record_openai_metrics(self, *, usage: dict[str, int] | None = None, rate_limits: dict[str, str] | None = None):
        if usage:
            self.last_openai_usage = {key: value for key, value in usage.items() if value is not None}
        if rate_limits:
            self.last_openai_rate_limits = {key: value for key, value in rate_limits.items() if value}

    def get_openai_usage_text(self) -> str:
        if not self.last_openai_usage:
            return ""

        input_tokens = self.last_openai_usage.get("input_tokens", 0)
        output_tokens = self.last_openai_usage.get("output_tokens", 0)
        total_tokens = self.last_openai_usage.get("total_tokens", 0)
        return (
            "Last OpenAI usage:\n"
            f"Input tokens: {input_tokens}\n"
            f"Output tokens: {output_tokens}\n"
            f"Total tokens: {total_tokens}"
        )

    def get_openai_rate_limit_text(self) -> str:
        if not self.last_openai_rate_limits:
            return ""

        lines = ["Last OpenAI rate-limit snapshot:"]
        for key in (
            "requests_limit",
            "requests_remaining",
            "requests_reset",
            "tokens_limit",
            "tokens_remaining",
            "tokens_reset",
        ):
            value = self.last_openai_rate_limits.get(key, "")
            if value:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(lines)

    def get_runtime_snapshot(self) -> dict:
        snapshot = asdict(self.state)
        snapshot["last_openai_usage"] = self.last_openai_usage
        snapshot["last_openai_rate_limits"] = self.last_openai_rate_limits
        return snapshot

    def get_active_llm_provider(self) -> str:
        return self.state.active_llm_provider

    def get_active_llm_model(self) -> str:
        return self.state.active_llm_model

    def get_active_image_provider(self) -> str:
        return self.state.active_image_provider

    def get_active_image_model(self) -> str:
        return self.state.active_image_model

    def get_effective_local_image_backend(self) -> str:
        if self.state.active_image_provider in {"automatic1111", "comfyui"}:
            return self.state.active_image_provider

        if self.state.active_image_provider == "local":
            if self._provider_ready("automatic1111"):
                return "automatic1111"
            if self._provider_ready("comfyui"):
                return "comfyui"
        return PREFERRED_LOCAL_IMAGE_BACKEND

    def _provider_ready(self, provider: str) -> bool:
        return bool(self.state.provider_status.get(provider, False))

    def _uses_local_backend(self) -> bool:
        return any(provider in LOCAL_IMAGE_PROVIDERS for provider in (self.state.active_llm_provider, self.state.active_image_provider))

    def _default_llm_model_for_provider(self, provider: str) -> str:
        mapping = {"openai": OPENAI_MODEL, "ollama": OLLAMA_MODEL, "hf": HF_MODEL}
        return mapping.get(provider, OPENAI_MODEL)
