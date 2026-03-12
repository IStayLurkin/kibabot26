from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from core.config import CUDA_PREFERRED, HF_MODEL, IMAGE_PROVIDER, LLM_PROVIDER, OLLAMA_MODEL, OPENAI_IMAGE_MODEL, OPENAI_MODEL
from core.logging_config import get_logger
from database.model_registry import find_models, get_model, get_runtime_settings, list_models, set_runtime_setting, upsert_model
from services.hardware_service import HardwareService

logger = get_logger(__name__)

LOCAL_MODEL_STORAGE_DIR = Path("bot_dl_storage")
LLM_DIR = LOCAL_MODEL_STORAGE_DIR / "llm"
IMAGE_DIR = LOCAL_MODEL_STORAGE_DIR / "image"
IMAGE_HINTS = ("sd", "sdxl", "flux", "stable-diffusion")


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
            active_device="CPU",
        )

    async def initialize(self):
        await self.register_default_models()
        await self.refresh_hardware_status()
        await self.scan_local_storage()
        await self.sync_models("llm")
        await self.sync_models("image")
        await self.load_runtime_settings()

    async def register_default_models(self):
        defaults = [
            ("openai", OPENAI_MODEL, "llm", "default", None, ["chat"], "openai", ""),
            ("openai", OPENAI_IMAGE_MODEL, "image", "default", None, ["image"], "openai", ""),
            ("ollama", OLLAMA_MODEL, "llm", "default", None, ["chat"], "ollama", "cuda"),
            ("hf", HF_MODEL, "llm", "default", None, ["chat"], "hf", ""),
        ]

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
        logger.info("[cuda_detected] enabled=%s gpu=%s", "yes" if self.state.cuda_enabled else "no", self.state.gpu_name or "unknown")
        self._apply_device_selection()
        return hardware

    def _apply_device_selection(self):
        preferred = "cuda" if self.state.cuda_preferred else self.state.preferred_compute_device
        if preferred == "cuda" and self.state.cuda_enabled:
            self.state.preferred_compute_device = "cuda"
            self.state.active_device = "CUDA"
        else:
            self.state.preferred_compute_device = "cpu"
            self.state.active_device = "CPU"

        logger.info("[device_selected] preferred=%s active=%s gpu=%s", self.state.preferred_compute_device.upper(), self.state.active_device, self.state.gpu_name or "none")

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
                capabilities=["chat"],
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
                backend="local",
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
                        capabilities=["chat"],
                        backend="ollama",
                        preferred_device=self.state.preferred_compute_device,
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
                        preferred_device=self.state.preferred_compute_device,
                        update_last_synced=True,
                    )
                    logger.info("[model_discovered] provider=ollama type=image model=%s", model["model_name"])
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

        return {"model_type": model_type, "discovered": discovered, "count": len(discovered)}

    async def add_model(self, provider: str, model_name: str, model_type: str):
        provider = provider.strip().lower()
        model_name = model_name.strip()
        await upsert_model(
            provider,
            model_name,
            model_type,
            source="manual",
            enabled=True,
            capabilities=["chat"] if model_type == "llm" else ["image"],
            backend=provider,
            preferred_device=self.state.preferred_compute_device if provider in {"local", "ollama"} else "",
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
            return f"Active LLM provider: {self.state.active_llm_provider}\nModel: {self.state.active_llm_model}"
        return f"Active image provider: {self.state.active_image_provider}\nModel: {self.state.active_image_model}"

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
        self._apply_device_selection()
        local_backend_active = any(
            provider in {"local", "ollama"}
            for provider in (self.state.active_llm_provider, self.state.active_image_provider)
        )

        lines = [
            f"CUDA status: {'enabled' if self.state.cuda_enabled else 'disabled'}",
            f"Preferred device: {self.state.preferred_compute_device.upper()}",
            f"Active device: {self.state.active_device}",
            f"GPU detected: {self.state.gpu_name or 'Not detected'}",
            f"Ollama available: {'yes' if self.state.ollama_available else 'no'}",
            f"Local backend using GPU: {'yes' if local_backend_active and self.state.active_device == 'CUDA' else 'no'}",
        ]

        if self.state.active_device == "CPU" and self.state.cuda_enabled:
            lines.append("Fallback: CPU only because the active backend or model does not currently support GPU execution.")
        elif self.state.active_device == "CPU":
            lines.append("Fallback: CPU only if needed.")

        return "\n".join(lines)

    def answer_natural_language_query(self, text: str) -> str | None:
        lowered = text.strip().lower()
        if lowered in {"what model are you using", "what llm are you using", "what model you using"}:
            return self.get_current_model_text("llm")
        if lowered in {"what image model are you using", "what image generator are you using"}:
            return self.get_current_model_text("image")
        if "3090 ti" in lowered:
            return f"GPU detected: {self.state.gpu_name or 'Not detected'}\nActive device: {self.state.active_device}"
        if "is ollama available" in lowered:
            return f"Ollama available: {'yes' if self.state.ollama_available else 'no'}"
        return None

    def get_runtime_snapshot(self) -> dict:
        return asdict(self.state)

    def get_active_llm_provider(self) -> str:
        return self.state.active_llm_provider

    def get_active_llm_model(self) -> str:
        return self.state.active_llm_model

    def get_active_image_provider(self) -> str:
        return self.state.active_image_provider

    def get_active_image_model(self) -> str:
        return self.state.active_image_model

    def _default_llm_model_for_provider(self, provider: str) -> str:
        mapping = {"openai": OPENAI_MODEL, "ollama": OLLAMA_MODEL, "hf": HF_MODEL}
        return mapping.get(provider, OPENAI_MODEL)
