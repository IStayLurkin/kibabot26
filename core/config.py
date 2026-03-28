import logging
import os
from dotenv import load_dotenv
from core.constants import BOT_DEFAULT_PREFIX

_config_logger = logging.getLogger(__name__)

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# --- HARDWARE PREFERENCE DEFINITION ---
CUDA_PREFERRED = os.getenv("CUDA_PREFERRED", "true").strip().lower() in {"1", "true", "yes", "on"}

# Pin device 0 without importing torch — torch checks CUDA_VISIBLE_DEVICES at first use
if CUDA_PREFERRED:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --- DRIVE REDIRECTION ---
os.environ["HF_HOME"] = os.getenv("HF_HOME", "J:/aistorage/huggingface_cache")
os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME", "J:/aistorage/torch_cache")
os.environ["OLLAMA_MODELS"] = os.getenv("OLLAMA_MODELS", "G:/ollamamodels")
os.environ["PIP_CACHE_DIR"] = os.getenv("PIP_CACHE_DIR", "J:/aistorage/pip_cache")

def _parse_int_list(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        _config_logger.warning("Invalid int config value %r, using default %d", value, default)
        return default


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        _config_logger.warning("Invalid float config value %r, using default %f", value, default)
        return default

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

BOT_PREFIX = os.getenv("BOT_PREFIX", BOT_DEFAULT_PREFIX)
BOT_TIMEZONE = os.getenv("BOT_TIMEZONE", "America/Los_Angeles")

PREFERRED_LOCAL_IMAGE_BACKEND = os.getenv("PREFERRED_LOCAL_IMAGE_BACKEND", "automatic1111").strip().lower()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_TEMPERATURE = _parse_float(os.getenv("LLM_TEMPERATURE", "0.8"), 0.8)
LLM_MAX_TOKENS = _parse_int(os.getenv("LLM_MAX_TOKENS", "2500"), 2500)
AGENTIC_CHAT_ENABLED = os.getenv("AGENTIC_CHAT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
AGENTIC_CHAT_MAX_TOKENS = _parse_int(os.getenv("AGENTIC_CHAT_MAX_TOKENS", "2500"), 2500)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:latest")
OLLAMA_CLI_PATH = os.getenv("OLLAMA_CLI_PATH", "").strip()
OLLAMA_REQUEST_TIMEOUT_SECONDS = _parse_int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "180"), 180)
OLLAMA_NUM_CTX = _parse_int(os.getenv("OLLAMA_NUM_CTX", "8192"), 8192)

# --- THINKING MODELS ---
THINKING_FAST_MODEL = os.getenv("THINKING_FAST_MODEL", "deepseek-r1:7b").strip()
THINKING_BEST_MODEL = os.getenv("THINKING_BEST_MODEL", "deepseek-r1:32b").strip()

# --- CODING MODELS ---
CODING_FAST_MODEL = os.getenv("CODING_FAST_MODEL", "qwen2.5-coder:7b").strip()
CODING_BEST_MODEL = os.getenv("CODING_BEST_MODEL", "devstral:24b").strip()

# --- VISION MODELS ---
VISION_FAST_MODEL = os.getenv("VISION_FAST_MODEL", "moondream").strip()
VISION_BEST_MODEL = os.getenv("VISION_BEST_MODEL", "llava:34b").strip()
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

# --- FISH SPEECH TTS ---
FISH_SPEECH_BASE_URL = os.getenv("FISH_SPEECH_BASE_URL", "http://localhost:8080").strip()
FISH_SPEECH_ENABLED = os.getenv("FISH_SPEECH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

# --- PARAKEET STT ---
PARAKEET_MODEL = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-1.1b").strip()
PARAKEET_ENABLED = os.getenv("PARAKEET_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

# --- MEM0 MEMORY ---
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "").strip()  # Leave empty for local mode
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:cerebras")

IMAGE_ENABLED = os.getenv("IMAGE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
VIDEO_ENABLED = os.getenv("VIDEO_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
CODEGEN_ENABLED = os.getenv("CODEGEN_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
OSINT_ENABLED = os.getenv("OSINT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

SAFE_OSINT_ONLY = os.getenv("SAFE_OSINT_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}

MEDIA_OUTPUT_DIR = os.getenv("MEDIA_OUTPUT_DIR", "J:/aistorage/generated_media")
MODEL_STORAGE_ROOT = os.getenv("MODEL_STORAGE_ROOT", "J:/aistorage/huggingface_cache")
MODEL_PULL_TIMEOUT_SECONDS = _parse_int(os.getenv("MODEL_PULL_TIMEOUT_SECONDS", "1800"), 1800)
ENABLED_MODEL_PROVIDERS = _parse_str_list(os.getenv("ENABLED_MODEL_PROVIDERS", "ollama,local,hf,automatic1111,comfyui"))
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", LLM_PROVIDER).strip().lower()

if DEFAULT_MODEL_PROVIDER not in ENABLED_MODEL_PROVIDERS:
    _config_logger.warning(
        "DEFAULT_MODEL_PROVIDER=%r is not in ENABLED_MODEL_PROVIDERS=%r. "
        "Falling back to first enabled provider.",
        DEFAULT_MODEL_PROVIDER,
        ENABLED_MODEL_PROVIDERS,
    )
    DEFAULT_MODEL_PROVIDER = ENABLED_MODEL_PROVIDERS[0] if ENABLED_MODEL_PROVIDERS else DEFAULT_MODEL_PROVIDER

CODE_WORKSPACE_ROOT = os.getenv("CODE_WORKSPACE_ROOT", "J:/aistorage/code_workspace")
CODE_SANDBOX_MODE = os.getenv("CODE_SANDBOX_MODE", "subprocess").strip().lower()
CODE_EXECUTION_TIMEOUT_SECONDS = _parse_int(os.getenv("CODE_EXECUTION_TIMEOUT_SECONDS", "20"), 20)
CODE_MAX_OUTPUT_CHARS = _parse_int(os.getenv("CODE_MAX_OUTPUT_CHARS", "6000"), 6000)
CODE_ALLOWED_USER_IDS = _parse_int_list(os.getenv("CODE_ALLOWED_USER_IDS", ""))
CODE_ALLOWED_ROLE_IDS = _parse_int_list(os.getenv("CODE_ALLOWED_ROLE_IDS", ""))

if not CODE_ALLOWED_USER_IDS and not CODE_ALLOWED_ROLE_IDS:
    _config_logger.warning(
        "CODE_ALLOWED_USER_IDS and CODE_ALLOWED_ROLE_IDS are both empty. "
        "Code execution will only be accessible to server admins."
    )

MAX_PROMPT_LENGTH = _parse_int(os.getenv("MAX_PROMPT_LENGTH", "1800"), 1800)
MAX_TTS_LENGTH = _parse_int(os.getenv("MAX_TTS_LENGTH", "1500"), 1500)
MAX_CODE_REQUEST_LENGTH = _parse_int(os.getenv("MAX_CODE_REQUEST_LENGTH", "4000"), 4000)

AGENT_DEFAULT_COOLDOWN_SECONDS = _parse_int(os.getenv("AGENT_DEFAULT_COOLDOWN_SECONDS", "8"), 8)
AGENT_MAX_CONTEXT_MESSAGES = _parse_int(os.getenv("AGENT_MAX_CONTEXT_MESSAGES", "15"), 15)

AUTOMATIC1111_BASE_URL = os.getenv("AUTOMATIC1111_BASE_URL", "").strip()
AUTOMATIC1111_DEFAULT_MODEL = os.getenv("AUTOMATIC1111_DEFAULT_MODEL", "automatic1111").strip()
AUTOMATIC1111_STEPS = _parse_int(os.getenv("AUTOMATIC1111_STEPS", "28"), 28)
AUTOMATIC1111_CFG_SCALE = _parse_float(os.getenv("AUTOMATIC1111_CFG_SCALE", "7.0"), 7.0)

COMFYUI_BASE_URL = os.getenv("COMFYUI_BASE_URL", "").strip()
COMFYUI_DEFAULT_MODEL = os.getenv("COMFYUI_DEFAULT_MODEL", "comfyui").strip()
COMFYUI_STEPS = _parse_int(os.getenv("COMFYUI_STEPS", "24"), 24)
COMFYUI_CFG_SCALE = _parse_float(os.getenv("COMFYUI_CFG_SCALE", "7.0"), 7.0)
COMFYUI_SAMPLER_NAME = os.getenv("COMFYUI_SAMPLER_NAME", "euler").strip()
COMFYUI_SCHEDULER = os.getenv("COMFYUI_SCHEDULER", "normal").strip()
COMFYUI_WIDTH = _parse_int(os.getenv("COMFYUI_WIDTH", "1024"), 1024)
COMFYUI_HEIGHT = _parse_int(os.getenv("COMFYUI_HEIGHT", "1024"), 1024)

IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", LLM_PROVIDER).strip().lower()
VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", LLM_PROVIDER).strip().lower()
VIDEO_PROVIDER = os.getenv("VIDEO_PROVIDER", LLM_PROVIDER).strip().lower()
DEFAULT_IMAGE_MODEL_PROVIDER = os.getenv("DEFAULT_IMAGE_MODEL_PROVIDER", IMAGE_PROVIDER).strip().lower()
MUSIC_PROVIDER = os.getenv("MUSIC_PROVIDER", "auto").strip().lower()
MUSIC_STUDIO_API_URL = os.getenv("MUSIC_STUDIO_API_URL", "").strip()
MUSIC_STUDIO_API_KEY = os.getenv("MUSIC_STUDIO_API_KEY", "").strip()
MUSIC_REQUEST_TIMEOUT_SECONDS = _parse_int(os.getenv("MUSIC_REQUEST_TIMEOUT_SECONDS", "180"), 180)
MUSIC_DEFAULT_QUALITY = os.getenv("MUSIC_DEFAULT_QUALITY", "studio").strip().lower()
MEDIA_SAFETY_MODE = os.getenv("MEDIA_SAFETY_MODE", "none").strip().lower()

GALLERY_CHANNEL_ID = os.getenv("GALLERY_CHANNEL_ID", "").strip()
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY", "").strip()
LOCAL_IMAGE_DIR = os.getenv("LOCAL_IMAGE_DIR", "").strip()
GPU_TOTAL_VRAM_MB = _parse_int(os.getenv("GPU_TOTAL_VRAM_MB", "24576"), 24576)

SEARXNG_ENABLED = os.getenv("SEARXNG_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080").strip()
SEARXNG_MAX_RESULTS = _parse_int(os.getenv("SEARXNG_MAX_RESULTS", "5"), 5)

YUE_REPO_PATH = os.getenv(
    "YUE_REPO_PATH",
    "G:/code/python/learn_python/bot/YuE/inference",  # YuE lives on G (code, not storage)
).strip()
