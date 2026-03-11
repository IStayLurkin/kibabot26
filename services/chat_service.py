from database.chat_memory import (
    get_recent_chat_messages,
    get_user_memory,
    get_conversation_summary,
)
from services.memory_service import format_memory
from services.chat_router import get_rule_based_fallback
from services.time_service import (
    is_date_time_question,
    build_current_datetime_reply,
)
from core.logging_config import get_logger
from core.constants import CHAT_RECENT_MESSAGE_LIMIT

logger = get_logger(__name__)


async def generate_dynamic_reply(
    llm,
    display_name: str,
    user_id: str,
    channel_id: str,
    session_id: int,
    user_text: str,
) -> str:
    if is_date_time_question(user_text):
        return build_current_datetime_reply(user_text, llm.timezone_name)

    recent_messages = await get_recent_chat_messages(session_id, limit=CHAT_RECENT_MESSAGE_LIMIT)
    memory_rows = await get_user_memory(user_id)
    memory = format_memory(memory_rows)
    conversation_summary = await get_conversation_summary(user_id, channel_id)

    try:
        reply = await llm.generate_reply(
            user_display_name=display_name,
            user_message=user_text,
            memory=memory,
            recent_messages=recent_messages,
            conversation_summary=conversation_summary,
        )
        if reply and reply.strip():
            return reply.strip()
    except Exception as exc:
        logger.exception("LLM error: %s", exc)

    return get_rule_based_fallback(
        display_name,
        user_text,
        memory=memory,
        recent_messages=recent_messages,
        conversation_summary=conversation_summary,
    )