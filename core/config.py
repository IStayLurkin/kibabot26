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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:cerebras")