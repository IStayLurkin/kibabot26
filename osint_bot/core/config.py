from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OSINT_DISCORD_BOT_TOKEN = os.getenv("OSINT_DISCORD_BOT_TOKEN")
OSINT_BOT_PREFIX = os.getenv("OSINT_BOT_PREFIX", "!")
OSINT_LLM_PROVIDER = os.getenv("OSINT_LLM_PROVIDER", "ollama").strip().lower()

OSINT_OLLAMA_BASE_URL = os.getenv("OSINT_OLLAMA_BASE_URL", "http://localhost:11434/v1")
OSINT_OLLAMA_API_KEY = os.getenv("OSINT_OLLAMA_API_KEY", "ollama")
OSINT_OLLAMA_MODEL = os.getenv("OSINT_OLLAMA_MODEL", "qwen3:8b")

OSINT_HF_BASE_URL = os.getenv("OSINT_HF_BASE_URL", "https://router.huggingface.co/v1")
OSINT_HF_TOKEN = os.getenv("OSINT_HF_TOKEN")
OSINT_HF_MODEL = os.getenv("OSINT_HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:cerebras")

OSINT_SAFE_MODE = os.getenv("OSINT_SAFE_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
OSINT_REQUIRE_AUTH_FOR_ACTIVE_CHECKS = os.getenv(
    "OSINT_REQUIRE_AUTH_FOR_ACTIVE_CHECKS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
OSINT_REQUEST_TIMEOUT_SECONDS = int(os.getenv("OSINT_REQUEST_TIMEOUT_SECONDS", "10"))
OSINT_MAX_OUTPUT_CHARS = int(os.getenv("OSINT_MAX_OUTPUT_CHARS", "1800"))
OSINT_HF_FALLBACK_ENABLED = os.getenv("OSINT_HF_FALLBACK_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

