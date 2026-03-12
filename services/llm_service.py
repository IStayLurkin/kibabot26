import asyncio
import json
import os
import time
from typing import Dict, List, Tuple

from openai import OpenAI

from core.config import (
    AGENTIC_CHAT_ENABLED,
    AGENTIC_CHAT_MAX_TOKENS,
    BOT_TIMEZONE,
    HF_BASE_URL,
    HF_MODEL,
    HF_TOKEN,
    IMAGE_PROVIDER,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_IMAGE_FORMAT,
    OPENAI_IMAGE_MODEL,
    OPENAI_IMAGE_QUALITY,
    OPENAI_IMAGE_SIZE,
    OPENAI_MODEL,
    OPENAI_TTS_MODEL,
    OPENAI_TTS_VOICE,
    VOICE_PROVIDER,
)
from core.logging_config import get_logger
from services.time_service import format_current_datetime_context

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are Kiba Bot, a Discord assistant with expense tracking features.

Core behavior:
- Be natural, concise, and helpful.
- Avoid canned filler and repetitive follow-up questions.
- Do not use embed-style formatting in normal chat replies.
- Use remembered user facts when relevant.
- Use recent conversation context when relevant.
- Use conversation summaries when relevant.
- For any current date or time question, use the provided runtime date/time context.
- Never guess the current date, day, month, year, or time from model memory.
- Do not mention internal prompts, SQL tables, or system architecture.

Agent policy:
- First infer what the user is actually trying to accomplish.
- Decide whether they need a direct answer, a plan, a clarifying question, or action-oriented help.
- If the user has a goal, optimize for moving them toward completion.
- Ask targeted clarifying questions only when the missing detail is blocking.
- When the request is actionable, give concrete next steps instead of generic encouragement.
- If a tool is available and relevant, shape the response around using that tool.
- Prefer concise responses unless the user asks for detail.
"""


def _extract_json_object(content: str) -> dict | None:
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


class LLMService:
    def __init__(self, performance_tracker=None):
        self.provider = LLM_PROVIDER
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.timezone_name = BOT_TIMEZONE
        self.agentic_chat_enabled = AGENTIC_CHAT_ENABLED
        self.agentic_chat_max_tokens = AGENTIC_CHAT_MAX_TOKENS
        self.performance_tracker = performance_tracker
        self._client_cache: dict[str, tuple[OpenAI, str]] = {}

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
        intent_category: str = "",
        conversation_goal: str = "",
        response_mode: str = "",
        tool_context: str = "",
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
            f"Detected intent: {intent_category or 'unknown'}\n"
            f"Active conversation goal: {conversation_goal or 'None'}\n"
            f"Requested response mode: {response_mode or 'direct_reply'}\n"
            f"Relevant tool context: {tool_context or 'None'}\n"
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
        cached = self._client_cache.get(provider)
        if cached is not None:
            return cached

        if provider == "openai":
            client_and_model = self._openai_client()
        elif provider == "ollama":
            client_and_model = self._ollama_client()
        elif provider == "hf":
            client_and_model = self._hf_client()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self._client_cache[provider] = client_and_model
        return client_and_model

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
        intent_category: str = "",
        conversation_goal: str = "",
        response_mode: str = "",
        tool_context: str = "",
    ) -> str:
        messages = self._build_messages(
            user_display_name=user_display_name,
            user_message=user_message,
            memory=memory,
            recent_messages=recent_messages,
            conversation_summary=conversation_summary,
            intent_category=intent_category,
            conversation_goal=conversation_goal,
            response_mode=response_mode,
            tool_context=tool_context,
        )

        errors = []
        providers = self._build_provider_chain()

        for provider in providers:
            started_at = time.perf_counter()
            try:
                client, model = self._get_client_and_model_for_provider(provider)

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if self.performance_tracker is not None:
                    self.performance_tracker.record_service_call(
                        f"llm.chat_completion.{provider}",
                        (time.perf_counter() - started_at) * 1000,
                    )

                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()

                errors.append(f"{provider}: empty response")

            except Exception as exc:
                if self.performance_tracker is not None:
                    self.performance_tracker.record_service_call(
                        f"llm.chat_completion.{provider}",
                        (time.perf_counter() - started_at) * 1000,
                    )
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
        intent_category: str = "",
        conversation_goal: str = "",
        response_mode: str = "",
        tool_context: str = "",
    ) -> str:
        started_at = time.perf_counter()
        try:
            return await asyncio.to_thread(
                self._generate_reply_sync,
                user_display_name,
                user_message,
                memory,
                recent_messages,
                conversation_summary,
                intent_category,
                conversation_goal,
                response_mode,
                tool_context,
            )
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "llm.generate_reply",
                    (time.perf_counter() - started_at) * 1000,
                )

    def _generate_agent_reply_sync(
        self,
        user_display_name: str,
        user_message: str,
        memory: Dict[str, str],
        recent_messages: List[Tuple[str, str, str]],
        conversation_summary: str = "",
        intent_category: str = "",
        conversation_goal: str = "",
        pending_question: str = "",
        tool_context: str = "",
    ) -> dict:
        messages = self._build_messages(
            user_display_name=user_display_name,
            user_message=user_message,
            memory=memory,
            recent_messages=recent_messages,
            conversation_summary=conversation_summary,
            intent_category=intent_category,
            conversation_goal=conversation_goal,
            response_mode="agentic",
            tool_context=tool_context,
        )

        plan_prompt = {
            "role": "system",
            "content": (
                "Analyze the user's goal before replying.\n"
                "Return strict JSON only.\n"
                "Schema:\n"
                "{\n"
                '  "intent": "casual_chat|question_answering|multi_step_help|planning|troubleshooting|tool_use_request|code_generation_analysis",\n'
                '  "goal": "short goal summary",\n'
                '  "response_mode": "direct|agentic|clarify",\n'
                '  "needs_clarification": true,\n'
                '  "clarifying_question": "one targeted question or empty string",\n'
                '  "tool_suggestion": "tool name or empty string",\n'
                '  "tool_reason": "short explanation or empty string",\n'
                '  "answer": "final user-facing reply text",\n'
                '  "next_steps": ["optional short next step", "optional short next step"],\n'
                '  "state_update": {\n'
                '    "goal": "updated goal or empty string",\n'
                '    "pending_question": "question still waiting on or empty string"\n'
                "  }\n"
                "}\n"
                "Rules:\n"
                "- Be goal-oriented and context-aware.\n"
                "- Ask a clarifying question only if a missing detail blocks useful progress.\n"
                "- If the user is trying to accomplish something, the answer should move them forward.\n"
                "- Keep the answer concise unless more detail is clearly needed.\n"
                f"- Existing pending question: {pending_question or 'None'}.\n"
            ),
        }
        messages.append(plan_prompt)

        raw = self._complete_messages_sync(
            messages,
            temperature=0.3,
            max_tokens=self.agentic_chat_max_tokens,
        )
        parsed = _extract_json_object(raw)

        if parsed is None:
            logger.warning("Agent planner returned non-JSON content: %s", raw)
            return {
                "intent": intent_category or "question_answering",
                "goal": conversation_goal or user_message[:120],
                "response_mode": "direct",
                "needs_clarification": False,
                "clarifying_question": "",
                "tool_suggestion": "",
                "tool_reason": "",
                "answer": raw.strip(),
                "next_steps": [],
                "state_update": {
                    "goal": conversation_goal or user_message[:120],
                    "pending_question": "",
                },
            }

        return parsed

    async def generate_agent_reply(
        self,
        user_display_name: str,
        user_message: str,
        memory: Dict[str, str],
        recent_messages: List[Tuple[str, str, str]],
        conversation_summary: str = "",
        intent_category: str = "",
        conversation_goal: str = "",
        pending_question: str = "",
        tool_context: str = "",
    ) -> dict:
        started_at = time.perf_counter()
        try:
            return await asyncio.to_thread(
                self._generate_agent_reply_sync,
                user_display_name,
                user_message,
                memory,
                recent_messages,
                conversation_summary,
                intent_category,
                conversation_goal,
                pending_question,
                tool_context,
            )
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "llm.generate_agent_reply",
                    (time.perf_counter() - started_at) * 1000,
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
                    "Keep important facts, preferences, ongoing topics, goals, and unresolved questions. "
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
            started_at = time.perf_counter()
            try:
                client, model = self._get_client_and_model_for_provider(provider)

                response = client.chat.completions.create(
                    model=model,
                    messages=summary_messages,
                    temperature=0.2,
                    max_tokens=180,
                )

                if self.performance_tracker is not None:
                    self.performance_tracker.record_service_call(
                        f"llm.summary_completion.{provider}",
                        (time.perf_counter() - started_at) * 1000,
                    )

                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()

                errors.append(f"{provider}: empty summary")

            except Exception as exc:
                if self.performance_tracker is not None:
                    self.performance_tracker.record_service_call(
                        f"llm.summary_completion.{provider}",
                        (time.perf_counter() - started_at) * 1000,
                    )
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
        started_at = time.perf_counter()
        try:
            return await asyncio.to_thread(
                self._generate_summary_sync,
                recent_messages,
                existing_summary,
            )
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "llm.generate_summary",
                    (time.perf_counter() - started_at) * 1000,
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
                    "{"
                    '"should_store": true or false, '
                    '"memory_key": "short_key_or_empty", '
                    '"memory_value": "value_or_empty"'
                    "}"
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
            started_at = time.perf_counter()
            try:
                client, model = self._get_client_and_model_for_provider(provider)

                response = client.chat.completions.create(
                    model=model,
                    messages=extraction_messages,
                    temperature=0.1,
                    max_tokens=120,
                )

                if self.performance_tracker is not None:
                    self.performance_tracker.record_service_call(
                        f"llm.memory_completion.{provider}",
                        (time.perf_counter() - started_at) * 1000,
                    )

                content = response.choices[0].message.content
                if not content or not content.strip():
                    errors.append(f"{provider}: empty extraction")
                    continue

                parsed = _extract_json_object(content)
                if isinstance(parsed, dict):
                    return parsed

                errors.append(f"{provider}: extraction was not a dict")

            except Exception as exc:
                if self.performance_tracker is not None:
                    self.performance_tracker.record_service_call(
                        f"llm.memory_completion.{provider}",
                        (time.perf_counter() - started_at) * 1000,
                    )
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
        started_at = time.perf_counter()
        try:
            return await asyncio.to_thread(
                self._extract_memory_sync,
                user_message,
                existing_memory,
            )
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "llm.extract_memory",
                    (time.perf_counter() - started_at) * 1000,
                )

    def _simple_messages(
        self,
        prompt: str,
        system_prompt: str = "You are Kiba Bot. Be helpful, accurate, and concise.",
    ) -> List[dict]:
        current_datetime_context = format_current_datetime_context(self.timezone_name)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": current_datetime_context},
            {"role": "user", "content": prompt},
        ]

    def _complete_messages_sync(
        self,
        messages: List[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        errors = []
        providers = self._build_provider_chain()

        for provider in providers:
            started_at = time.perf_counter()
            try:
                client, model = self._get_client_and_model_for_provider(provider)

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                )

                if self.performance_tracker is not None:
                    self.performance_tracker.record_service_call(
                        f"llm.complete_messages.{provider}",
                        (time.perf_counter() - started_at) * 1000,
                    )

                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()

                errors.append(f"{provider}: empty response")

            except Exception as exc:
                if self.performance_tracker is not None:
                    self.performance_tracker.record_service_call(
                        f"llm.complete_messages.{provider}",
                        (time.perf_counter() - started_at) * 1000,
                    )
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
                continue

        raise RuntimeError("All LLM providers failed | " + " | ".join(errors))

    async def complete_messages(
        self,
        messages: List[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        started_at = time.perf_counter()
        try:
            return await asyncio.to_thread(
                self._complete_messages_sync,
                messages,
                temperature,
                max_tokens,
            )
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "llm.complete_messages",
                    (time.perf_counter() - started_at) * 1000,
                )

    async def generate_text(self, prompt: str) -> str:
        messages = self._simple_messages(prompt)
        return await self.complete_messages(messages)

    async def generate_response(self, prompt: str) -> str:
        return await self.generate_text(prompt)

    async def chat(self, prompt: str) -> str:
        return await self.generate_text(prompt)

    def _get_client_and_model_for_media_provider(self, provider: str, media_type: str) -> tuple[OpenAI, str]:
        if provider == "openai":
            client, _model = self._get_client_and_model_for_provider("openai")
            if media_type == "image":
                return client, OPENAI_IMAGE_MODEL
            if media_type == "voice":
                return client, OPENAI_TTS_MODEL
            return client, self._get_active_model_name()

        if provider == "ollama":
            raise RuntimeError(f"{media_type.title()} generation is not supported for ollama in this build.")

        if provider == "hf":
            raise RuntimeError(f"{media_type.title()} generation is not supported for hf in this build.")

        raise ValueError(f"Unsupported media provider: {provider}")

    async def generate_image(self, prompt: str) -> dict:
        started_at = time.perf_counter()
        try:
            return await asyncio.to_thread(self._generate_image_sync, prompt)
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "llm.generate_image",
                    (time.perf_counter() - started_at) * 1000,
                )

    def _generate_image_sync(self, prompt: str) -> dict:
        provider = IMAGE_PROVIDER
        client, model = self._get_client_and_model_for_media_provider(provider, "image")

        try:
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=OPENAI_IMAGE_SIZE,
                quality=OPENAI_IMAGE_QUALITY,
                output_format=OPENAI_IMAGE_FORMAT,
                response_format="b64_json",
            )
        except Exception as exc:
            raise RuntimeError(f"Image generation failed via {provider}: {exc}") from exc

        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError("Image generation returned no data.")

        first = data[0]

        b64_json = getattr(first, "b64_json", None)
        if b64_json:
            return {"b64_json": b64_json}

        image_base64 = getattr(first, "image_base64", None)
        if image_base64:
            return {"image_base64": image_base64}

        url = getattr(first, "url", None)
        if url:
            return {"url": url}

        raise RuntimeError("Unsupported image response format.")

    async def text_to_speech(self, text: str) -> bytes:
        started_at = time.perf_counter()
        try:
            return await asyncio.to_thread(self._text_to_speech_sync, text)
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "llm.text_to_speech",
                    (time.perf_counter() - started_at) * 1000,
                )

    def _text_to_speech_sync(self, text: str) -> bytes:
        provider = VOICE_PROVIDER
        client, model = self._get_client_and_model_for_media_provider(provider, "voice")

        try:
            response = client.audio.speech.create(
                model=model,
                voice=OPENAI_TTS_VOICE,
                input=text,
            )
        except Exception as exc:
            raise RuntimeError(f"TTS failed via {provider}: {exc}") from exc

        if hasattr(response, "read"):
            return response.read()

        if hasattr(response, "content") and isinstance(response.content, bytes):
            return response.content

        if isinstance(response, bytes):
            return response

        raise RuntimeError("Unsupported TTS response format.")
