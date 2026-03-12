import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

from core.config import (
    AGENTIC_CHAT_ENABLED,
    AGENTIC_CHAT_MAX_TOKENS,
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_STEPS,
    BOT_TIMEZONE,
    COMFYUI_BASE_URL,
    COMFYUI_CFG_SCALE,
    COMFYUI_DEFAULT_MODEL,
    COMFYUI_HEIGHT,
    COMFYUI_SAMPLER_NAME,
    COMFYUI_SCHEDULER,
    COMFYUI_STEPS,
    COMFYUI_WIDTH,
    HF_BASE_URL,
    HF_MODEL,
    HF_TOKEN,
    IMAGE_PROVIDER,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    MEDIA_OUTPUT_DIR,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_IMAGE_MODEL,
    OPENAI_IMAGE_QUALITY,
    OPENAI_IMAGE_SIZE,
    OPENAI_MODEL,
    OPENAI_TTS_MODEL,
    OPENAI_TTS_VOICE,
    OPENAI_VIDEO_MODEL,
    OPENAI_VIDEO_SECONDS,
    OPENAI_VIDEO_SIZE,
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
    def __init__(self, performance_tracker=None, model_runtime_service=None):
        self.provider = LLM_PROVIDER
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.timezone_name = BOT_TIMEZONE
        self.agentic_chat_enabled = AGENTIC_CHAT_ENABLED
        self.agentic_chat_max_tokens = AGENTIC_CHAT_MAX_TOKENS
        self.performance_tracker = performance_tracker
        self.model_runtime_service = model_runtime_service
        self._client_cache: dict[str, OpenAI] = {}
        self.media_output_dir = Path(MEDIA_OUTPUT_DIR)
        self.media_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_active_model_name(self) -> str:
        if self.model_runtime_service is not None:
            self.provider = self.model_runtime_service.get_active_llm_provider()
            return self.model_runtime_service.get_active_llm_model()
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

    def _openai_client(self) -> OpenAI:
        return OpenAI(api_key=OPENAI_API_KEY)

    def _ollama_client(self) -> OpenAI:
        return OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
            timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )

    def _hf_client(self) -> OpenAI:
        return OpenAI(
            base_url=HF_BASE_URL,
            api_key=HF_TOKEN,
        )

    def _get_client_for_provider(self, provider: str) -> OpenAI:
        cached = self._client_cache.get(provider)
        if cached is not None:
            return cached

        if provider == "openai":
            client = self._openai_client()
        elif provider == "ollama":
            client = self._ollama_client()
        elif provider == "hf":
            client = self._hf_client()
        elif provider in {"local", "automatic1111", "comfyui"}:
            raise RuntimeError(f"{provider} does not use the OpenAI SDK client path.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self._client_cache[provider] = client
        return client

    def _build_provider_chain(self) -> List[str]:
        if self.model_runtime_service is not None:
            active_provider = self.model_runtime_service.get_active_llm_provider()
            self.provider = active_provider
            return [active_provider]

        chains = {
            "openai": ["openai", "hf", "ollama"],
            "hf": ["hf", "openai", "ollama"],
            "ollama": ["ollama", "openai", "hf"],
        }
        return chains.get(self.provider, ["openai", "hf", "ollama"])

    def _get_model_for_provider(self, provider: str, media_type: str = "llm") -> str:
        if self.model_runtime_service is not None:
            if media_type == "image":
                return self.model_runtime_service.get_active_image_model()
            return self.model_runtime_service.get_active_llm_model()

        if media_type == "image":
            return OPENAI_IMAGE_MODEL
        if provider == "openai":
            return OPENAI_MODEL
        if provider == "ollama":
            return OLLAMA_MODEL
        if provider == "hf":
            return HF_MODEL
        return OPENAI_MODEL

    def _extract_usage(self, parsed_response) -> dict[str, int]:
        usage = getattr(parsed_response, "usage", None)
        if usage is None:
            return {}

        return {
            "input_tokens": getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0,
            "output_tokens": getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0,
            "total_tokens": getattr(usage, "total_tokens", None) or 0,
        }

    def _extract_rate_limits(self, headers) -> dict[str, str]:
        if not headers:
            return {}

        mapping = {
            "requests_limit": headers.get("x-ratelimit-limit-requests", ""),
            "requests_remaining": headers.get("x-ratelimit-remaining-requests", ""),
            "requests_reset": headers.get("x-ratelimit-reset-requests", ""),
            "tokens_limit": headers.get("x-ratelimit-limit-tokens", ""),
            "tokens_remaining": headers.get("x-ratelimit-remaining-tokens", ""),
            "tokens_reset": headers.get("x-ratelimit-reset-tokens", ""),
        }
        return {key: value for key, value in mapping.items() if value}

    def _record_openai_metrics(self, parsed_response=None, headers=None):
        if self.model_runtime_service is None:
            return

        usage = self._extract_usage(parsed_response) if parsed_response is not None else {}
        rate_limits = self._extract_rate_limits(headers)
        if usage or rate_limits:
            self.model_runtime_service.record_openai_metrics(usage=usage, rate_limits=rate_limits)

    def _create_chat_completion(self, provider: str, *, model: str, messages: List[dict], temperature: float, max_tokens: int):
        client = self._get_client_for_provider(provider)
        if provider == "openai":
            raw_response = client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed_response = raw_response.parse()
            self._record_openai_metrics(parsed_response, raw_response.headers)
            return parsed_response

        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _post_json(self, url: str, payload: dict, *, timeout: int = 60) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "KibaBot/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    def _get_json(self, url: str, *, timeout: int = 60) -> dict:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "KibaBot/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Request failed for {url}: {exc}") from exc

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
                model = self._get_model_for_provider(provider, "llm")
                response = self._create_chat_completion(
                    provider,
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
                model = self._get_model_for_provider(provider, "llm")
                response = self._create_chat_completion(
                    provider,
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
                model = self._get_model_for_provider(provider, "llm")
                response = self._create_chat_completion(
                    provider,
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
                model = self._get_model_for_provider(provider, "llm")
                response = self._create_chat_completion(
                    provider,
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
        if self.model_runtime_service is not None:
            provider = self.model_runtime_service.get_active_image_provider()

        if provider == "openai":
            client = self._get_client_for_provider("openai")
            try:
                raw_response = client.images.with_raw_response.generate(
                    model=self._get_model_for_provider("openai", "image"),
                    prompt=prompt,
                    size=OPENAI_IMAGE_SIZE,
                    quality=OPENAI_IMAGE_QUALITY,
                    output_format="png",
                )
                response = raw_response.parse()
                self._record_openai_metrics(response, raw_response.headers)
            except Exception as exc:
                raise RuntimeError(f"Image generation failed via openai: {exc}") from exc

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

            raise RuntimeError("Unsupported OpenAI image response format.")

        if provider in {"automatic1111", "local"}:
            backend = provider
            if provider == "local" and self.model_runtime_service is not None:
                backend = self.model_runtime_service.get_effective_local_image_backend()
            if backend == "automatic1111":
                return self._generate_image_automatic1111(prompt)
            if backend == "comfyui":
                return self._generate_image_comfyui(prompt)
            raise RuntimeError("No supported local image backend is configured.")

        if provider == "comfyui":
            return self._generate_image_comfyui(prompt)

        if provider == "ollama":
            raise RuntimeError("Ollama image generation is registered in the runtime, but no compatible image-generation endpoint is wired yet.")

        if provider == "hf":
            raise RuntimeError("Hugging Face image generation is not wired in this build yet.")

        raise RuntimeError(f"Unsupported image provider: {provider}")

    def _generate_image_automatic1111(self, prompt: str) -> dict:
        if not AUTOMATIC1111_BASE_URL:
            raise RuntimeError("AUTOMATIC1111_BASE_URL is not configured.")

        model_name = self._get_model_for_provider("automatic1111", "image")
        payload = {
            "prompt": prompt,
            "steps": AUTOMATIC1111_STEPS,
            "cfg_scale": AUTOMATIC1111_CFG_SCALE,
            "sampler_name": "Euler a",
            "width": COMFYUI_WIDTH,
            "height": COMFYUI_HEIGHT,
            "override_settings": {"sd_model_checkpoint": model_name},
        }
        response = self._post_json(f"{AUTOMATIC1111_BASE_URL.rstrip('/')}/sdapi/v1/txt2img", payload, timeout=240)
        images = response.get("images", [])
        if not images:
            raise RuntimeError("Automatic1111 returned no images.")
        return {"image_base64": images[0]}

    def _generate_image_comfyui(self, prompt: str) -> dict:
        if not COMFYUI_BASE_URL:
            raise RuntimeError("COMFYUI_BASE_URL is not configured.")

        model_name = self._get_model_for_provider("comfyui", "image") or COMFYUI_DEFAULT_MODEL
        workflow = {
            "1": {
                "inputs": {"ckpt_name": model_name},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {"text": prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": "", "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"width": COMFYUI_WIDTH, "height": COMFYUI_HEIGHT, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "5": {
                "inputs": {
                    "seed": int(time.time() * 1000) % 2147483647,
                    "steps": COMFYUI_STEPS,
                    "cfg": COMFYUI_CFG_SCALE,
                    "sampler_name": COMFYUI_SAMPLER_NAME,
                    "scheduler": COMFYUI_SCHEDULER,
                    "denoise": 1,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                },
                "class_type": "KSampler",
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            "7": {
                "inputs": {"filename_prefix": "kiba", "images": ["6", 0]},
                "class_type": "SaveImage",
            },
        }

        submission = self._post_json(
            f"{COMFYUI_BASE_URL.rstrip('/')}/prompt",
            {"prompt": workflow, "client_id": f"kiba-{uuid.uuid4().hex}"},
            timeout=60,
        )
        prompt_id = submission.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("ComfyUI did not return a prompt_id.")

        history = self._poll_comfyui_history(prompt_id)
        outputs = history.get(prompt_id, {}).get("outputs", {})
        for node_output in outputs.values():
            images = node_output.get("images", [])
            if not images:
                continue
            image = images[0]
            filename = image.get("filename")
            subfolder = image.get("subfolder", "")
            image_type = image.get("type", "output")
            if filename:
                query = urllib.parse.urlencode(
                    {"filename": filename, "subfolder": subfolder, "type": image_type}
                )
                url = f"{COMFYUI_BASE_URL.rstrip('/')}/view?{query}"
                return {"url": url}

        raise RuntimeError("ComfyUI completed the prompt but returned no downloadable image.")

    def _poll_comfyui_history(self, prompt_id: str) -> dict:
        history_url = f"{COMFYUI_BASE_URL.rstrip('/')}/history/{prompt_id}"
        last_error = None
        for _attempt in range(240):
            try:
                history = self._get_json(history_url, timeout=30)
                if history and prompt_id in history:
                    return history
            except Exception as exc:
                last_error = exc
            time.sleep(1.0)

        if last_error is not None:
            raise RuntimeError(f"ComfyUI history polling failed: {last_error}") from last_error
        raise RuntimeError("Timed out waiting for ComfyUI image generation.")

    async def generate_video(self, prompt: str) -> dict:
        started_at = time.perf_counter()
        try:
            return await asyncio.to_thread(self._generate_video_sync, prompt)
        finally:
            if self.performance_tracker is not None:
                self.performance_tracker.record_service_call(
                    "llm.generate_video",
                    (time.perf_counter() - started_at) * 1000,
                )

    def _generate_video_sync(self, prompt: str) -> dict:
        client = self._get_client_for_provider("openai")
        try:
            raw_response = client.videos.with_raw_response.create(
                model=OPENAI_VIDEO_MODEL,
                prompt=prompt,
                seconds=OPENAI_VIDEO_SECONDS,
                size=OPENAI_VIDEO_SIZE,
            )
            created_video = raw_response.parse()
            self._record_openai_metrics(created_video, raw_response.headers)
            video = client.videos.poll(created_video.id)
        except Exception as exc:
            raise RuntimeError(f"Video generation failed via openai: {exc}") from exc

        if getattr(video, "status", "") != "completed":
            last_error = getattr(video, "last_error", None)
            raise RuntimeError(f"Video generation did not complete successfully. Status: {getattr(video, 'status', 'unknown')} | Error: {last_error}")

        try:
            content = client.videos.download_content(video.id)
            video_bytes = content.read()
        except Exception as exc:
            raise RuntimeError(f"Video content download failed: {exc}") from exc

        filename = f"video_{uuid.uuid4().hex}.mp4"
        path = self.media_output_dir / filename
        path.write_bytes(video_bytes)
        return {"file_path": str(path)}

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
        if provider != "openai":
            raise RuntimeError(f"Voice generation via {provider} is not wired in this build.")

        client = self._get_client_for_provider("openai")

        try:
            response = client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
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
