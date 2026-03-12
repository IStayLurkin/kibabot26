import os
from dotenv import load_dotenv
from core.constants import BOT_DEFAULT_PREFIX

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

BOT_PREFIX = os.getenv("BOT_PREFIX", BOT_DEFAULT_PREFIX)
BOT_TIMEZONE = os.getenv("BOT_TIMEZONE", "America/Los_Angeles")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "220"))
AGENTIC_CHAT_ENABLED = os.getenv("AGENTIC_CHAT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
AGENTIC_CHAT_MAX_TOKENS = int(os.getenv("AGENTIC_CHAT_MAX_TOKENS", "500"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

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

MEDIA_OUTPUT_DIR = os.getenv("MEDIA_OUTPUT_DIR", "generated_media")

MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "1800"))
MAX_TTS_LENGTH = int(os.getenv("MAX_TTS_LENGTH", "1500"))
MAX_CODE_REQUEST_LENGTH = int(os.getenv("MAX_CODE_REQUEST_LENGTH", "4000"))

AGENT_DEFAULT_COOLDOWN_SECONDS = int(os.getenv("AGENT_DEFAULT_COOLDOWN_SECONDS", "8"))
AGENT_MAX_CONTEXT_MESSAGES = int(os.getenv("AGENT_MAX_CONTEXT_MESSAGES", "15"))

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", LLM_PROVIDER).strip().lower()
VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", LLM_PROVIDER).strip().lower()
VIDEO_PROVIDER = os.getenv("VIDEO_PROVIDER", LLM_PROVIDER).strip().lower()
