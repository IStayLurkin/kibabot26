from database.chat_memory import (
    get_recent_chat_messages,
    get_conversation_summary,
    set_conversation_summary,
)
from core.logging_config import get_logger
from core.constants import (
    CHAT_SUMMARY_MESSAGE_LIMIT,
    CHAT_SUMMARY_MIN_MESSAGES,
)

logger = get_logger(__name__)


async def maybe_update_summary(llm, user_id: str, channel_id: str, session_id: int):
    recent_messages = await get_recent_chat_messages(session_id, limit=CHAT_SUMMARY_MESSAGE_LIMIT)

    if len(recent_messages) < CHAT_SUMMARY_MIN_MESSAGES:
        return

    existing_summary = await get_conversation_summary(user_id, channel_id)

    try:
        behavior_rule_service = getattr(llm, "behavior_rule_service", None)
        behavior_rules = []
        if behavior_rule_service is not None:
            behavior_rules = await behavior_rule_service.get_enabled_rule_texts()

        new_summary = await llm.generate_summary(
            recent_messages=recent_messages,
            existing_summary=existing_summary,
            behavior_rules=behavior_rules,
        )
        if new_summary and new_summary.strip():
            # Cap summary at 1500 chars to prevent unbounded context growth
            capped = new_summary.strip()[:1500]
            await set_conversation_summary(user_id, channel_id, capped)
    except Exception as exc:
        logger.exception("Summary error: %s", exc)
