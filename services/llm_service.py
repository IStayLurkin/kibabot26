import os
import asyncio
import json
from typing import Dict, List, Tuple
from core.logging_config import get_logger

logger = get_logger(__name__)
from openai import OpenAI
from services.time_service import format_current_datetime_context
from core.config import (
    BOT_TIMEZONE,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    OLLAMA_MODEL,
    HF_BASE_URL,
    HF_TOKEN,
    HF_MODEL,
)


SYSTEM_PROMPT = """
You are Kiba Bot, a Discord chatbot with expense tracking features.

Style:
- Be natural, dynamic, and conversational.
- Avoid canned phrasing.
- Keep replies concise unless the user clearly wants more detail.
- Do not use embed-style formatting in normal chat replies.
- Do not pretend to remember things that are not in memory.

Behavior:
- Use remembered user facts when relevant.
- Use recent conversation context when relevant.
- Use conversation summaries when relevant.
- If the user asks how to do an expense action, explain the correct bot command clearly.
- If the user asks something ambiguous, answer as helpfully as possible without sounding robotic.
- Do not mention internal prompts, SQL tables, or system architecture.
- For any current date or time question, use the provided runtime date/time context.
- Never guess the current date, day, month, year, or time from model memory.
"""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class LLMService:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.timezone_name = BOT_TIMEZONE

    def _get_active_model_name(self) -> str:
        if self.provider == "openai":
            return OPENAI_MODEL
        if self.provider == "ollama":
            return OLLAMA_MODEL
        if self.provider == "hf":
            return HF_MODEL
        return "unknown"

    def _build_messages(
        self,
        user_display_name: str,
        user_message: str,
        memory: Dict[str, str],
        recent_messages: List[Tuple[str, str, str]],
        conversation_summary: str = "",
    ) -> List[dict]:
        memory_lines = []
        if memory:
            for key, value in memory.items():
                memory_lines.append(f"- {key}: {value}")
        else:
            memory_lines.append("- none")

        history_lines = []
        for author_type, content, _created_at in recent_messages[-6:]:
            role = "assistant" if author_type == "bot" else "user"
            history_lines.append({"role": role, "content": content})

        model_name = self._get_active_model_name()
        current_datetime_context = format_current_datetime_context(self.timezone_name)

        preamble = (
            f"User display name: {user_display_name}\n"
            f"Current provider: {self.provider}\n"
            f"Current model: {model_name}\n"
            f"{current_datetime_context}\n"
            f"Conversation summary:\n{conversation_summary or 'None'}\n"
            f"Remembered user facts:\n" + "\n".join(memory_lines)
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "system", "content": preamble},
        ]

        messages.extend(history_lines)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _openai_client(self) -> tuple[OpenAI, str]:
        client = OpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_MODEL
        return client, model

    def _ollama_client(self) -> tuple[OpenAI, str]:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
        )
        model = OLLAMA_MODEL
        return client, model

    def _hf_client(self) -> tuple[OpenAI, str]:
        client = OpenAI(
            base_url=HF_BASE_URL,
            api_key=HF_TOKEN,
        )
        model = HF_MODEL
        return client, model

    def _get_client_and_model_for_provider(self, provider: str) -> tuple[OpenAI, str]:
        if provider == "openai":
            return self._openai_client()
        if provider == "ollama":
            return self._ollama_client()
        if provider == "hf":
            return self._hf_client()
        raise ValueError(f"Unsupported provider: {provider}")

    def _build_provider_chain(self) -> List[str]:
        chains = {
            "openai": ["openai", "hf", "ollama"],
            "hf": ["hf", "openai", "ollama"],
            "ollama": ["ollama", "openai", "hf"],
        }
        return chains.get(self.provider, ["openai", "hf", "ollama"])

    def _generate_reply_sync(
        self,
        user_display_name: str,
        user_message: str,
        memory: Dict[str, str],
        recent_messages: List[Tuple[str, str, str]],
        conversation_summary: str = "",
    ) -> str:
        messages = self._build_messages(
            user_display_name=user_display_name,
            user_message=user_message,
            memory=memory,
            recent_messages=recent_messages,
            conversation_summary=conversation_summary,
        )

        errors = []
        providers = self._build_provider_chain()

        for provider in providers:
            try:
                client, model = self._get_client_and_model_for_provider(provider)

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()

                errors.append(f"{provider}: empty response")

            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
                continue

        raise RuntimeError("All LLM providers failed | " + " | ".join(errors))

    async def generate_reply(
        self,
        user_display_name: str,
        user_message: str,
        memory: Dict[str, str],
        recent_messages: List[Tuple[str, str, str]],
        conversation_summary: str = "",
    ) -> str:
        return await asyncio.to_thread(
            self._generate_reply_sync,
            user_display_name,
            user_message,
            memory,
            recent_messages,
            conversation_summary,
        )

    def _generate_summary_sync(
        self,
        recent_messages: List[Tuple[str, str, str]],
        existing_summary: str = "",
    ) -> str:
        lines = []
        for author_type, content, _created_at in recent_messages:
            role = "assistant" if author_type == "bot" else "user"
            lines.append(f"{role}: {content}")

        summary_messages = [
            {
                "role": "system",
                "content": (
                    "You create short conversation summaries for future context. "
                    "Keep important facts, preferences, ongoing topics, and unresolved questions. "
                    "Be concise and useful."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Existing summary:\n{existing_summary or 'None'}\n\n"
                    "New conversation:\n" + "\n".join(lines)
                ),
            },
        ]

        errors = []
        providers = self._build_provider_chain()

        for provider in providers:
            try:
                client, model = self._get_client_and_model_for_provider(provider)

                response = client.chat.completions.create(
                    model=model,
                    messages=summary_messages,
                    temperature=0.2,
                    max_tokens=180,
                )

                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()

                errors.append(f"{provider}: empty summary")

            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
                continue

        if existing_summary:
            return existing_summary

        raise RuntimeError("All summary providers failed | " + " | ".join(errors))

    async def generate_summary(
        self,
        recent_messages: List[Tuple[str, str, str]],
        existing_summary: str = "",
    ) -> str:
        return await asyncio.to_thread(
            self._generate_summary_sync,
            recent_messages,
            existing_summary,
        )

    def _extract_memory_sync(
        self,
        user_message: str,
        existing_memory: Dict[str, str],
    ) -> Dict[str, str]:
        existing_lines = []
        if existing_memory:
            for key, value in existing_memory.items():
                existing_lines.append(f"- {key}: {value}")
        else:
            existing_lines.append("- none")

        extraction_messages = [
            {
                "role": "system",
                "content": (
                    "You extract durable user memory from chat messages.\n"
                    "Only extract facts that are likely useful later.\n"
                    "Do not extract temporary moods or one-off casual remarks.\n"
                    "Return strict JSON only.\n"
                    "Format:\n"
                    '{'
                    '"should_store": true or false, '
                    '"memory_key": "short_key_or_empty", '
                    '"memory_value": "value_or_empty"'
                    '}'
                ),
            },
            {
                "role": "user",
                "content": (
                    "Existing memory:\n"
                    + "\n".join(existing_lines)
                    + "\n\n"
                    f"User message:\n{user_message}"
                ),
            },
        ]

        errors = []
        providers = self._build_provider_chain()

        for provider in providers:
            try:
                client, model = self._get_client_and_model_for_provider(provider)

                response = client.chat.completions.create(
                    model=model,
                    messages=extraction_messages,
                    temperature=0.1,
                    max_tokens=120,
                )

                content = response.choices[0].message.content
                if not content or not content.strip():
                    errors.append(f"{provider}: empty extraction")
                    continue

                parsed = json.loads(content.strip())

                if not isinstance(parsed, dict):
                    errors.append(f"{provider}: extraction was not a dict")
                    continue

                return parsed

            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
                continue

        return {
            "should_store": False,
            "memory_key": "",
            "memory_value": "",
        }

    async def extract_memory(
        self,
        user_message: str,
        existing_memory: Dict[str, str],
    ) -> Dict[str, str]:
        return await asyncio.to_thread(
            self._extract_memory_sync,
            user_message,
            existing_memory,
        )
