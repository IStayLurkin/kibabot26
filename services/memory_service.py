from database.chat_memory import get_user_memory, set_user_memory
from core.logging_config import get_logger

logger = get_logger(__name__)


def extract_memory_fact(text: str):
    cleaned = text.strip()
    lowered = cleaned.lower()

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


async def maybe_extract_ai_memory(llm, user_id: str, content: str):
    memory_rows = await get_user_memory(user_id)
    existing_memory = format_memory(memory_rows)

    try:
        extracted = await llm.extract_memory(
            user_message=content,
            existing_memory=existing_memory,
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

    await set_user_memory(user_id, memory_key, memory_value)
    return memory_key, memory_value


async def store_memory_if_found(llm, user_id: str, content: str):
    explicit_memory = extract_memory_fact(content)
    if explicit_memory:
        key, value = explicit_memory
        await set_user_memory(user_id, key, value)
        return key, value

    return await maybe_extract_ai_memory(llm, user_id, content)
