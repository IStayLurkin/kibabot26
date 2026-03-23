from database.chat_memory import get_user_memory, set_user_memory
from core.logging_config import get_logger
from services.tool_router import (
    INTENT_CASUAL_CHAT,
    INTENT_MULTI_STEP_HELP,
    INTENT_PLANNING,
    INTENT_TOOL_USE_REQUEST,
    INTENT_TROUBLESHOOTING,
    ToolRouter,
)

logger = get_logger(__name__)

_router = ToolRouter()

NON_MEMORY_PREFIXES = (
    "help me ",
    "can you help me",
    "how do i ",
    "plan ",
    "debug ",
    "fix ",
    "whois ",
    "dns ",
    "generate ",
    "make ",
    "create ",
    "show me ",
)

TASK_CONTEXT_MARKERS = (
    "income",
    "budget",
    "groceries",
    "rent",
    "fun money",
    "extra",
    "allocate",
    "monthly",
)


def extract_memory_fact(text: str):
    cleaned = text.strip()
    lowered = cleaned.lower()

    if "do not use emojis unless requested" in lowered:
        return "emoji_preference", "Do not use emojis unless requested."

    if "stop sending emojis" in lowered or "don't use emojis" in lowered or "do not use emojis" in lowered:
        return "emoji_preference", "Do not use emojis unless requested."

    if "my name is " in lowered:
        start = lowered.find("my name is ") + len("my name is ")
        return "name", cleaned[start:].strip()

    if lowered.startswith("remember that "):
        return "note", cleaned[14:].strip()

    if lowered.startswith("remember "):
        return "note", cleaned[9:].strip()

    if lowered.startswith("i prefer "):
        return "preference", cleaned[9:].strip()

    return None


def format_memory(memory_rows):
    return {key: value for key, value in memory_rows}


def should_attempt_memory_storage(content: str) -> bool:
    cleaned = content.strip()
    lowered = cleaned.lower()

    if not cleaned or len(cleaned.split()) < 2:
        return False

    explicit_memory = extract_memory_fact(cleaned)
    if explicit_memory:
        return True

    if any(lowered.startswith(prefix) for prefix in NON_MEMORY_PREFIXES):
        return False

    route_decision = _router.route(cleaned)

    if route_decision.intent in {
        INTENT_MULTI_STEP_HELP,
        INTENT_PLANNING,
        INTENT_TROUBLESHOOTING,
        INTENT_TOOL_USE_REQUEST,
    }:
        return False

    if route_decision.intent == INTENT_CASUAL_CHAT:
        return False

    blocked_phrases = (
        "budget",
        "plan",
        "steps",
        "error",
        "bug",
        "traceback",
        "how do i",
        "what should i do",
        "can you help",
    )
    if any(phrase in lowered for phrase in blocked_phrases):
        return False

    return True


async def maybe_extract_ai_memory(llm, user_id: str, content: str):
    if not should_attempt_memory_storage(content):
        return None

    memory_rows = await get_user_memory(user_id)
    existing_memory = format_memory(memory_rows)
    lowered = content.strip().lower()

    if any(marker in lowered for marker in TASK_CONTEXT_MARKERS):
        return None

    try:
        behavior_rule_service = getattr(llm, "behavior_rule_service", None)
        behavior_rules = []
        if behavior_rule_service is not None:
            behavior_rules = await behavior_rule_service.get_enabled_rule_texts()

        extracted = await llm.extract_memory(
            user_message=content,
            existing_memory=existing_memory,
            behavior_rules=behavior_rules,
        )
    except Exception as exc:
        logger.exception("Memory extraction error: %s", exc)
        return None

    if not isinstance(extracted, dict):
        return None

    should_store = extracted.get("should_store", False)
    memory_key = str(extracted.get("memory_key", "")).strip()
    memory_value = str(extracted.get("memory_value", "")).strip()

    if not should_store:
        return None

    if not memory_key or not memory_value:
        return None

    blocked_memory_keys = {
        "income",
        "salary",
        "budget",
        "budget_categories",
        "groceries",
        "rent",
        "fun_money",
        "extra",
    }
    if memory_key.lower() in blocked_memory_keys:
        return None

    if len(memory_value.split()) > 20:
        logger.debug("Skipping memory storage: value too long (%d words)", len(memory_value.split()))
        return None

    await set_user_memory(user_id, memory_key, memory_value)
    return memory_key, memory_value


async def store_memory_if_found(llm, user_id: str, content: str):
    explicit_memory = extract_memory_fact(content)
    if explicit_memory:
        key, value = explicit_memory
        await set_user_memory(user_id, key, value)
        return key, value

    return await maybe_extract_ai_memory(llm, user_id, content)


async def maybe_store_episodic_memory(
    llm,
    vector_memory_service,
    db,
    user_id: str,
    user_message: str,
    bot_reply: str,
) -> None:
    """
    After a chat turn, ask the LLM if anything is worth storing as an episodic memory.
    Runs as a background task — never raises.
    """
    if len(user_message.split()) < 3:
        return
    try:
        result = await llm.extract_episodic_memory(
            user_message=user_message,
            bot_reply=bot_reply,
        )
        if not isinstance(result, dict) or not result.get("should_store"):
            return
        content = str(result.get("content", "")).strip()
        if not content:
            return
        await vector_memory_service.store(db, user_id=user_id, content=content)
    except Exception as exc:
        logger.warning("[episodic_memory] Extraction failed: %s", exc)
