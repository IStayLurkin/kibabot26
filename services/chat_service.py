from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from core.constants import CHAT_RECENT_MESSAGE_LIMIT
from core.logging_config import get_logger
from database.chat_memory import (
    get_conversation_state,
    get_conversation_summary,
    get_recent_chat_messages,
    get_user_memory,
    set_conversation_state,
)
from database.db_connection import get_db
from services.chat_router import get_rule_based_fallback
from services.memory_service import format_memory, maybe_store_episodic_memory
from services.time_service import (
    build_current_datetime_reply,
    is_date_time_question,
)
from services.tool_router import (
    INTENT_CODE_GENERATION_ANALYSIS,
    INTENT_TOOL_USE_REQUEST,
    ToolRouter,
)

logger = get_logger(__name__)

_router = ToolRouter()


@dataclass(slots=True)
class ChatReply:
    content: str
    file_paths: list[str] = field(default_factory=list)
    intent: str = ""
    response_mode: str = "direct"
    goal: str = ""
    tool_name: str = ""


_CASUAL_PATTERNS = {
    "hey", "hi", "hello", "sup", "what's up", "whats up", "yo", "hiya",
    "yeah", "ye", "yep", "yup", "ok", "okay", "k", "lol", "lmao", "haha",
    "nice", "cool", "interesting", "wild", "damn", "bruh", "bro", "man",
    "nah", "nope", "no", "yes", "sure", "alright", "alr", "bet",
    "not much", "nm", "same", "fr", "facts", "true", "word",
}

def _should_retrieve_memory(text: str) -> bool:
    """Only retrieve vector memories when the message is substantive enough to warrant it.
    Skip retrieval for casual greetings, short affirmations, and vague statements
    that would cause semantic noise and inject irrelevant past topics."""
    stripped = text.strip().lower().rstrip("!?.,")
    # Skip if the whole message is a known casual phrase
    if stripped in _CASUAL_PATTERNS:
        return False
    # Skip very short messages (under 6 words) that are likely casual
    words = stripped.split()
    if len(words) < 6:
        return False
    return True


def _build_tool_context(route_decision, conversation_state: dict) -> str:
    parts = []

    if route_decision.tool_name:
        parts.append(f"requested tool: {route_decision.tool_name}")
    if route_decision.tool_input:
        parts.append(f"tool input: {route_decision.tool_input}")
    if conversation_state.get("last_tool"):
        parts.append(f"last tool used: {conversation_state['last_tool']}")
    if conversation_state.get("pending_question"):
        parts.append(f"pending question: {conversation_state['pending_question']}")

    return " | ".join(parts)


def _compose_agent_answer(plan: dict) -> str:
    answer = str(plan.get("answer", "")).strip()
    next_steps = plan.get("next_steps", [])

    if not isinstance(next_steps, list):
        next_steps = []

    cleaned_steps = [str(step).strip() for step in next_steps if str(step).strip()]
    if cleaned_steps:
        answer = answer.rstrip()
        answer += "\n\nNext steps:\n" + "\n".join(f"- {step}" for step in cleaned_steps[:3])

    return answer.strip()


def _record_tracker_duration(services: dict | None, name: str, started_at: float) -> None:
    services = services or {}
    llm = services.get("llm")
    tracker = getattr(llm, "performance_tracker", None)
    if tracker is None:
        return

    tracker.record_service_call(name, (time.perf_counter() - started_at) * 1000)


async def _run_tool(tool_name: str, tool_input: str, services: dict | None) -> ChatReply | None:
    services = services or {}
    cleaned_input = tool_input.strip()

    if tool_name == "osint":
        osint_service = services.get("osint_service")
        if osint_service is None:
            return ChatReply(
                content="I can do domain intelligence lookups, but the OSINT service is not available right now.",
                intent=INTENT_TOOL_USE_REQUEST,
                response_mode="tool",
                goal=cleaned_input,
                tool_name="osint",
            )

        lowered = cleaned_input.lower()
        if "dns" in lowered:
            result = await osint_service.dns_lookup(cleaned_input)
        elif "ssl" in lowered:
            result = await osint_service.ssl_lookup(cleaned_input)
        elif "whois" in lowered or "rdap" in lowered:
            domain = cleaned_input.split()[-1]
            result = await osint_service.whois_lookup(domain)
        elif "." in cleaned_input and " " not in cleaned_input:
            whois_result = await osint_service.whois_lookup(cleaned_input)
            dns_result = await osint_service.dns_lookup(cleaned_input)
            result = f"{whois_result}\n\n{dns_result}"
        else:
            result = await osint_service.lookup_query(cleaned_input)

        return ChatReply(
            content=result,
            intent=INTENT_TOOL_USE_REQUEST,
            response_mode="tool",
            goal=cleaned_input,
            tool_name="osint",
        )

    if tool_name == "code":
        codegen_service = services.get("codegen_service")
        if codegen_service is None:
            return ChatReply(
                content="I can help with code, but the code service is not available right now.",
                intent=INTENT_CODE_GENERATION_ANALYSIS,
                response_mode="tool",
                goal=cleaned_input,
                tool_name="code",
            )

        result = await codegen_service.generate_code_help(cleaned_input)
        return ChatReply(
            content=result,
            intent=INTENT_CODE_GENERATION_ANALYSIS,
            response_mode="tool",
            goal=cleaned_input,
            tool_name="code",
        )

    if tool_name == "image":
        image_service = services.get("image_service")
        if image_service is None:
            return ChatReply(
                content="I can generate images, but the image service is not available right now.",
                intent=INTENT_TOOL_USE_REQUEST,
                response_mode="tool",
                goal=cleaned_input,
                tool_name="image",
            )

        image_path = await image_service.generate_image(cleaned_input)
        return ChatReply(
            content="Here’s the image I generated.",
            file_paths=[image_path],
            intent=INTENT_TOOL_USE_REQUEST,
            response_mode="tool",
            goal=cleaned_input,
            tool_name="image",
        )

    if tool_name == "voice":
        voice_service = services.get("voice_service")
        if voice_service is None:
            return ChatReply(
                content="I can generate audio, but the voice service is not available right now.",
                intent=INTENT_TOOL_USE_REQUEST,
                response_mode="tool",
                goal=cleaned_input,
                tool_name="voice",
            )

        audio_path = await voice_service.text_to_speech(cleaned_input)
        return ChatReply(
            content="Here’s the audio version.",
            file_paths=[audio_path],
            intent=INTENT_TOOL_USE_REQUEST,
            response_mode="tool",
            goal=cleaned_input,
            tool_name="voice",
        )

    if tool_name == "video":
        # Try real backends first (animatediff = lowest VRAM, fastest)
        animatediff_service = services.get("animatediff_service")
        if animatediff_service is not None:
            video_path = await animatediff_service.generate(prompt=cleaned_input, callback=None)
            if video_path:
                return ChatReply(
                    content="Here’s the generated video.",
                    file_paths=[video_path],
                    intent=INTENT_TOOL_USE_REQUEST,
                    response_mode="tool",
                    goal=cleaned_input,
                    tool_name="video",
                )
            return ChatReply(
                content="Video generation failed. Check VRAM availability.",
                intent=INTENT_TOOL_USE_REQUEST,
                response_mode="tool",
                goal=cleaned_input,
                tool_name="video",
            )

        return ChatReply(
            content="Use !animatediff, !cogvideo2b, !cogvideo5b, or !wan for video generation.",
            intent=INTENT_TOOL_USE_REQUEST,
            response_mode="tool",
            goal=cleaned_input,
            tool_name="video",
        )

    if tool_name == "music":
        music_service = services.get("music_service")
        if music_service is None:
            return ChatReply(
                content="I can generate melodies, but the music service is not available right now.",
                intent=INTENT_TOOL_USE_REQUEST,
                response_mode="tool",
                goal=cleaned_input,
                tool_name="music",
            )

        melody_path = await music_service.generate_melody(cleaned_input)
        if not melody_path:
            return ChatReply(
                content="❌ Melody generation failed. Check VRAM availability.",
                intent=INTENT_TOOL_USE_REQUEST,
                response_mode="tool",
                goal=cleaned_input,
                tool_name="music",
            )
        return ChatReply(
            content="Here’s the melody I generated.",
            file_paths=[melody_path],
            intent=INTENT_TOOL_USE_REQUEST,
            response_mode="tool",
            goal=cleaned_input,
            tool_name="music",
        )

    return None


async def generate_dynamic_reply(
    llm,
    display_name: str,
    user_id: str,
    channel_id: str,
    session_id: int,
    user_text: str,
    services: dict | None = None,
) -> ChatReply:
    started_at = time.perf_counter()
    services = services or {}
    services.setdefault("llm", llm)
    logger.debug("[chat] generate_dynamic_reply called user_text=%r", user_text[:80])

    try:
        model_runtime_service = services.get("model_runtime_service")
        command_help_service = services.get("command_help_service")
        behavior_rule_service = services.get("behavior_rule_service")
        bot = services.get("bot")

        behavior_rules = []
        if behavior_rule_service is not None:
            behavior_rules = await behavior_rule_service.get_enabled_rule_texts()
            lowered = user_text.strip().lower()

            if lowered in {"what are the rules", "show the rules", "list the rules", "what rules do you have"}:
                return ChatReply(
                    content=await behavior_rule_service.get_rules_text(),
                    intent="question_answering",
                    response_mode="direct",
                    goal="list persistent behavior rules",
                    tool_name="behavior_rule",
                )

            if behavior_rule_service.looks_like_rule_request(user_text):
                rule_text = behavior_rule_service.extract_rule_text(user_text)
                _ok, message = await behavior_rule_service.add_rule(rule_text, created_by=user_id)
                return ChatReply(
                    content=message,
                    intent="tool_use_request",
                    response_mode="direct",
                    goal="set persistent behavior rule",
                    tool_name="behavior_rule",
                )

            if behavior_rule_service.looks_like_rule_edit_request(user_text):
                old_rule_text, new_rule_text = behavior_rule_service.extract_rule_replacement(user_text)
                _ok, message = await behavior_rule_service.replace_rule(
                    old_rule_text,
                    new_rule_text,
                    created_by=user_id,
                )
                return ChatReply(
                    content=message,
                    intent="tool_use_request",
                    response_mode="direct",
                    goal="edit persistent behavior rule",
                    tool_name="behavior_rule",
                )

        from database.behavior_rules_repository import get_bot_config
        from services.llm_service import PERSONALITIES, DEFAULT_PERSONALITY
        _user_personality = await get_bot_config(f"user_personality:{user_id}", "")
        _global_personality = await get_bot_config("active_personality", DEFAULT_PERSONALITY)
        active_personality = _user_personality if _user_personality in PERSONALITIES else (
            _global_personality if _global_personality in PERSONALITIES else DEFAULT_PERSONALITY
        )

        if model_runtime_service is not None:
            runtime_answer = model_runtime_service.answer_natural_language_query(user_text)
            if runtime_answer is not None:
                return ChatReply(
                    content=runtime_answer,
                    intent="question_answering",
                    response_mode="direct",
                    goal="answer runtime model question",
                )

            lowered = user_text.strip().lower()
            if lowered in {"use it", "switch to it", "turn it on", "use ollama"}:
                if model_runtime_service.get_last_runtime_topic() == "ollama_available":
                    return ChatReply(
                        content=await model_runtime_service.activate_ollama_default(),
                        intent="tool_use_request",
                        response_mode="direct",
                        goal="switch active llm to ollama",
                        tool_name="model_runtime",
                    )

            if lowered in {"why?", "why", "how come?", "how come", "what do you mean?", "what do you mean"}:
                runtime_reason = model_runtime_service.get_last_runtime_reason()
                if runtime_reason:
                    return ChatReply(
                        content=runtime_reason,
                        intent="question_answering",
                        response_mode="direct",
                        goal="answer runtime follow-up question",
                    )

            if lowered in {"are you using cuda", "are you using gpu", "what device are you using", "are you using my gpu"}:
                return ChatReply(
                    content=await model_runtime_service.get_hardware_status_text(),
                    intent="question_answering",
                    response_mode="direct",
                    goal="answer runtime hardware question",
                )

            if "what models are available" in lowered or "what image generators are available" in lowered:
                model_type = "image" if "image" in lowered or "generator" in lowered else "llm"
                return ChatReply(
                    content=await model_runtime_service.get_model_list_text(model_type),
                    intent="question_answering",
                    response_mode="direct",
                    goal="list runtime models",
                )

            if "what audio models are available" in lowered or "what voice models are available" in lowered or "what tts models are available" in lowered:
                return ChatReply(
                    content=await model_runtime_service.get_model_list_text("audio"),
                    intent="question_answering",
                    response_mode="direct",
                    goal="list runtime audio models",
                )

            if "what local models are available" in lowered:
                models = await model_runtime_service.get_models("llm")
                local_models = [model for model in models if model["provider"] == "local"]
                if not local_models:
                    content = "No local LLM models are registered right now."
                else:
                    content = "Local LLM models:\n" + "\n".join(f"- {model['model_name']}" for model in local_models)
                return ChatReply(
                    content=content,
                    intent="question_answering",
                    response_mode="direct",
                    goal="list local llm models",
                )

        if command_help_service is not None and bot is not None and command_help_service.matches_natural_language_help(user_text):
            lowered = user_text.strip().lower()
            if "what can you do" in lowered or "capabilities" in lowered or "abilities" in lowered:
                hidden_sections = set()
                if any("talk about expenses unless asked" in rule.lower() for rule in behavior_rules):
                    hidden_sections.update({"Expenses", "Budgets"})
                return ChatReply(
                    content=await command_help_service.build_capabilities_summary(bot, hidden_sections=hidden_sections),
                    intent="question_answering",
                    response_mode="direct",
                    goal="describe bot capabilities",
                )
            return ChatReply(
                content=await command_help_service.build_command_overview(bot),
                intent="question_answering",
                response_mode="direct",
                goal="list available commands",
            )

        if is_date_time_question(user_text):
            return ChatReply(
                content=build_current_datetime_reply(user_text, llm.timezone_name),
                intent="question_answering",
                response_mode="direct",
                goal="answer current date/time question",
            )

        recent_messages = await get_recent_chat_messages(session_id, limit=CHAT_RECENT_MESSAGE_LIMIT)
        memory_rows = await get_user_memory(user_id)
        memory = format_memory(memory_rows)

        lowered = user_text.lower().strip()
        if any(phrase in lowered for phrase in (
            "what do you remember", "what do you know about me",
            "do you remember me", "what do you remember about me",
            "do you remember anything about me",
        )):
            conversation_summary = await get_conversation_summary(user_id, channel_id)
            parts = [f"{k}: {v}" for k, v in memory.items()]
            if conversation_summary:
                parts.append(f"summary: {conversation_summary}")
            content = ("Here's what I have on you: " + "; ".join(parts)) if parts else "Nothing stored yet."
            return ChatReply(content=content, intent="question_answering", response_mode="direct", goal="recall memory")
        conversation_summary = await get_conversation_summary(user_id, channel_id)

        # Semantic memory retrieval
        relevant_memories = []
        vector_memory_service = (services or {}).get("vector_memory_service")
        mem0_service = (services or {}).get("mem0_service")
        active_memory_service = vector_memory_service
        db_conn = None
        try:
            from database.behavior_rules_repository import get_bot_config
            memory_mode = await get_bot_config("memory_mode", "local")
            active_memory_service = mem0_service if (memory_mode == "mem0" and mem0_service is not None) else vector_memory_service
            if active_memory_service is not None and _should_retrieve_memory(user_text):
                db_conn = await get_db()
                relevant_memories = await active_memory_service.retrieve(db_conn, user_id=user_id, query=user_text)
        except Exception as exc:
            logger.warning("[vector_memory] Retrieval failed in chat_service: %s", exc)

        conversation_state = await get_conversation_state(user_id, channel_id)

        route_decision = _router.route(user_text)
        conversation_goal = conversation_state.get("goal") or user_text.strip()
        tool_context = _build_tool_context(route_decision, conversation_state)

        if route_decision.tool_name:
            if route_decision.should_ask_clarifying_question:
                question = f"What exactly do you want me to do with the {route_decision.tool_name} tool?"
                await set_conversation_state(
                    user_id,
                    channel_id,
                    goal=conversation_goal,
                    last_intent=route_decision.intent,
                    response_mode="clarify",
                    last_tool=route_decision.tool_name,
                    pending_question=question,
                )
                return ChatReply(
                    content=question,
                    intent=route_decision.intent,
                    response_mode="clarify",
                    goal=conversation_goal,
                    tool_name=route_decision.tool_name,
                )

            try:
                tool_reply = await _run_tool(route_decision.tool_name, route_decision.tool_input, services)
                if tool_reply is not None:
                    await set_conversation_state(
                        user_id,
                        channel_id,
                        goal=tool_reply.goal or conversation_goal,
                        last_intent=tool_reply.intent or route_decision.intent,
                        response_mode=tool_reply.response_mode,
                        last_tool=route_decision.tool_name,
                        pending_question="",
                    )
                    return tool_reply
            except Exception as exc:
                logger.exception("Tool execution error: %s", exc)

        if route_decision.requires_agent and getattr(llm, "agentic_chat_enabled", False):
            try:
                plan = await llm.generate_agent_reply(
                    user_display_name=display_name,
                    user_message=user_text,
                    memory=memory,
                    recent_messages=recent_messages,
                    conversation_summary=conversation_summary,
                    intent_category=route_decision.intent,
                    conversation_goal=conversation_goal,
                    pending_question=conversation_state.get("pending_question", ""),
                    tool_context=tool_context,
                    behavior_rules=behavior_rules,
                    relevant_memories=relevant_memories,
                )

                state_update = plan.get("state_update", {})
                if not isinstance(state_update, dict):
                    state_update = {}

                if plan.get("needs_clarification"):
                    clarification_question = str(plan.get("clarifying_question", "")).strip() or "What part do you want help with first?"
                    await set_conversation_state(
                        user_id,
                        channel_id,
                        goal=str(state_update.get("goal", "")).strip() or conversation_goal,
                        last_intent=str(plan.get("intent", route_decision.intent)).strip(),
                        response_mode="clarify",
                        last_tool=str(plan.get("tool_suggestion", route_decision.tool_name)).strip(),
                        pending_question=clarification_question,
                    )
                    return ChatReply(
                        content=clarification_question,
                        intent=str(plan.get("intent", route_decision.intent)).strip(),
                        response_mode="clarify",
                        goal=str(state_update.get("goal", "")).strip() or conversation_goal,
                        tool_name=str(plan.get("tool_suggestion", route_decision.tool_name)).strip(),
                    )

                answer = _compose_agent_answer(plan)
                updated_goal = str(state_update.get("goal", "")).strip() or str(plan.get("goal", "")).strip() or conversation_goal
                pending_question = str(state_update.get("pending_question", "")).strip()
                tool_name = str(plan.get("tool_suggestion", route_decision.tool_name)).strip()
                response_mode = str(plan.get("response_mode", "agentic")).strip() or "agentic"
                intent = str(plan.get("intent", route_decision.intent)).strip() or route_decision.intent

                await set_conversation_state(
                    user_id,
                    channel_id,
                    goal=updated_goal,
                    last_intent=intent,
                    response_mode=response_mode,
                    last_tool=tool_name,
                    pending_question=pending_question,
                )

                if active_memory_service is not None and db_conn is not None:
                    asyncio.create_task(maybe_store_episodic_memory(
                        llm=llm,
                        vector_memory_service=active_memory_service,
                        db=db_conn,
                        user_id=user_id,
                        user_message=user_text,
                        bot_reply=answer,
                    ))

                return ChatReply(
                    content=answer,
                    intent=intent,
                    response_mode=response_mode,
                    goal=updated_goal,
                    tool_name=tool_name,
                )
            except Exception as exc:
                logger.exception("Agentic reply error: %s", exc)

        logger.debug("[chat] falling through to generate_reply")
        try:
            reply = await llm.generate_reply(
                user_display_name=display_name,
                user_message=user_text,
                memory=memory,
                recent_messages=recent_messages,
                conversation_summary=conversation_summary,
                intent_category=route_decision.intent,
                conversation_goal=conversation_goal,
                response_mode="direct",
                tool_context=tool_context,
                behavior_rules=behavior_rules,
                relevant_memories=relevant_memories,
                personality=active_personality,
            )
            if reply and reply.strip():
                await set_conversation_state(
                    user_id,
                    channel_id,
                    goal=conversation_goal,
                    last_intent=route_decision.intent,
                    response_mode="direct",
                    last_tool=route_decision.tool_name,
                    pending_question="",
                )
                if active_memory_service is not None and db_conn is not None:
                    asyncio.create_task(
                        maybe_store_episodic_memory(
                            llm=llm,
                            vector_memory_service=active_memory_service,
                            db=db_conn,
                            user_id=user_id,
                            user_message=user_text,
                            bot_reply=reply.strip(),
                        )
                    )
                return ChatReply(
                    content=reply.strip(),
                    intent=route_decision.intent,
                    response_mode="direct",
                    goal=conversation_goal,
                    tool_name=route_decision.tool_name,
                )
        except Exception as exc:
            logger.warning("Reply fallback: %s", exc)

        fallback = get_rule_based_fallback(
            display_name,
            user_text,
            memory=memory,
            recent_messages=recent_messages,
            conversation_summary=conversation_summary,
        )
        await set_conversation_state(
            user_id,
            channel_id,
            goal=conversation_goal,
            last_intent=route_decision.intent,
            response_mode="fallback",
            last_tool=route_decision.tool_name,
            pending_question="",
        )
        return ChatReply(
            content=fallback,
            intent=route_decision.intent,
            response_mode="fallback",
            goal=conversation_goal,
            tool_name=route_decision.tool_name,
        )
    finally:
        _record_tracker_duration(services, "chat.generate_dynamic_reply", started_at)
