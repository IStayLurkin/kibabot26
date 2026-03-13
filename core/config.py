import os
from dotenv import load_dotenv
from core.constants import BOT_DEFAULT_PREFIX

load_dotenv()
os.environ["HF_HOME"] = "G:/huggingface_cache"
os.environ["TORCH_HOME"] = "G:/torch_cache"
os.environ["OLLAMA_MODELS"] = "G:/ollamamodels"
os.environ["PIP_CACHE_DIR"] = "G:/pip_cache"

# Force CUDA context for the 3090 Ti
if CUDA_PREFERRED:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

BOT_PREFIX = os.getenv("BOT_PREFIX", BOT_DEFAULT_PREFIX)
BOT_TIMEZONE = os.getenv("BOT_TIMEZONE", "America/Los_Angeles")
CUDA_PREFERRED = os.getenv("CUDA_PREFERRED", "true").strip().lower() in {"1", "true", "yes", "on"}
PREFERRED_LOCAL_IMAGE_BACKEND = os.getenv("PREFERRED_LOCAL_IMAGE_BACKEND", "automatic1111").strip().lower()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "220"))
AGENTIC_CHAT_ENABLED = os.getenv("AGENTIC_CHAT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
AGENTIC_CHAT_MAX_TOKENS = int(os.getenv("AGENTIC_CHAT_MAX_TOKENS", "500"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:latest")
OLLAMA_CLI_PATH = os.getenv("OLLAMA_CLI_PATH", "").strip()
OLLAMA_REQUEST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "180"))

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

MEDIA_OUTPUT_DIR = os.getenv("MEDIA_OUTPUT_DIR", "G:/code/python/learn_python/bot/discord_bot_things/generated_media")
MODEL_STORAGE_ROOT = os.getenv("MODEL_STORAGE_ROOT", "G:/huggingface_cache")
MODEL_PULL_TIMEOUT_SECONDS = int(os.getenv("MODEL_PULL_TIMEOUT_SECONDS", "1800"))
ENABLED_MODEL_PROVIDERS = _parse_str_list(os.getenv("ENABLED_MODEL_PROVIDERS", "openai,ollama,local,hf,automatic1111,comfyui"))
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", LLM_PROVIDER).strip().lower()

CODE_WORKSPACE_ROOT = os.getenv("CODE_WORKSPACE_ROOT", "G:/code/python/learn_python/bot/discord_bot_things/code_workspace")
CODE_SANDBOX_MODE = os.getenv("CODE_SANDBOX_MODE", "subprocess").strip().lower()
CODE_EXECUTION_TIMEOUT_SECONDS = int(os.getenv("CODE_EXECUTION_TIMEOUT_SECONDS", "20"))
CODE_MAX_OUTPUT_CHARS = int(os.getenv("CODE_MAX_OUTPUT_CHARS", "6000"))
CODE_ALLOWED_USER_IDS = _parse_int_list(os.getenv("CODE_ALLOWED_USER_IDS", ""))
CODE_ALLOWED_ROLE_IDS = _parse_int_list(os.getenv("CODE_ALLOWED_ROLE_IDS", ""))

MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "1800"))
MAX_TTS_LENGTH = int(os.getenv("MAX_TTS_LENGTH", "1500"))
MAX_CODE_REQUEST_LENGTH = int(os.getenv("MAX_CODE_REQUEST_LENGTH", "4000"))

AGENT_DEFAULT_COOLDOWN_SECONDS = int(os.getenv("AGENT_DEFAULT_COOLDOWN_SECONDS", "8"))
AGENT_MAX_CONTEXT_MESSAGES = int(os.getenv("AGENT_MAX_CONTEXT_MESSAGES", "15"))

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "medium")
OPENAI_IMAGE_FORMAT = os.getenv("OPENAI_IMAGE_FORMAT", "png")
OPENAI_VIDEO_MODEL = os.getenv("OPENAI_VIDEO_MODEL", "sora-2")
OPENAI_VIDEO_SIZE = os.getenv("OPENAI_VIDEO_SIZE", "720x1280")
OPENAI_VIDEO_SECONDS = int(os.getenv("OPENAI_VIDEO_SECONDS", "4"))
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

AUTOMATIC1111_BASE_URL = os.getenv("AUTOMATIC1111_BASE_URL", "").strip()
AUTOMATIC1111_DEFAULT_MODEL = os.getenv("AUTOMATIC1111_DEFAULT_MODEL", "automatic1111").strip()
AUTOMATIC1111_STEPS = int(os.getenv("AUTOMATIC1111_STEPS", "28"))
AUTOMATIC1111_CFG_SCALE = float(os.getenv("AUTOMATIC1111_CFG_SCALE", "7.0"))

COMFYUI_BASE_URL = os.getenv("COMFYUI_BASE_URL", "").strip()
COMFYUI_DEFAULT_MODEL = os.getenv("COMFYUI_DEFAULT_MODEL", "comfyui").strip()
COMFYUI_STEPS = int(os.getenv("COMFYUI_STEPS", "24"))
COMFYUI_CFG_SCALE = float(os.getenv("COMFYUI_CFG_SCALE", "7.0"))
COMFYUI_SAMPLER_NAME = os.getenv("COMFYUI_SAMPLER_NAME", "euler").strip()
COMFYUI_SCHEDULER = os.getenv("COMFYUI_SCHEDULER", "normal").strip()
COMFYUI_WIDTH = int(os.getenv("COMFYUI_WIDTH", "1024"))
COMFYUI_HEIGHT = int(os.getenv("COMFYUI_HEIGHT", "1024"))

IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", LLM_PROVIDER).strip().lower()
VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", LLM_PROVIDER).strip().lower()
VIDEO_PROVIDER = os.getenv("VIDEO_PROVIDER", LLM_PROVIDER).strip().lower()
DEFAULT_IMAGE_MODEL_PROVIDER = os.getenv("DEFAULT_IMAGE_MODEL_PROVIDER", IMAGE_PROVIDER).strip().lower()
MUSIC_PROVIDER = os.getenv("MUSIC_PROVIDER", "auto").strip().lower()
MUSIC_STUDIO_API_URL = os.getenv("MUSIC_STUDIO_API_URL", "").strip()
MUSIC_STUDIO_API_KEY = os.getenv("MUSIC_STUDIO_API_KEY", "").strip()
MUSIC_REQUEST_TIMEOUT_SECONDS = int(os.getenv("MUSIC_REQUEST_TIMEOUT_SECONDS", "180"))
MUSIC_DEFAULT_QUALITY = os.getenv("MUSIC_DEFAULT_QUALITY", "studio").strip().lower()
MEDIA_SAFETY_MODE = os.getenv("MEDIA_SAFETY_MODE", "none").strip().lower()
