import re


def is_greeting(text: str) -> bool:
    return bool(re.fullmatch(r"(?:hello|hi|hey|yo|sup|what's up|whats up)", text.strip()))


def is_thanks(text: str) -> bool:
    thanks_words = ["thanks", "thank you", "ty"]
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in thanks_words)


def is_help_request(text: str) -> bool:
    help_words = ["help", "commands", "what can you do", "what capabilities do you have", "capabilities"]
    return any(word in text for word in help_words)


def is_expense_topic(text: str) -> bool:
    expense_words = ["expense", "expenses", "spending", "money", "budget"]
    return any(word in text for word in expense_words)


def is_private_info_request(text: str) -> bool:
    private_info_phrases = [
        "what is my ip",
        "what's my ip",
        "tell me my ip",
        "what is my address",
        "what's my address",
        "what is my location",
        "where am i",
        "what device am i on",
    ]
    return any(phrase in text for phrase in private_info_phrases)


def get_help_response() -> str:
    return "Use `!help` or `!commands` to see what I can do."


def get_expense_response(text: str) -> str:
    add_words = ["add", "log", "record", "save"]
    list_words = ["list", "show", "view", "see"]
    recent_words = ["recent", "latest", "last"]
    total_words = ["total", "sum", "overall"]
    category_words = ["category", "categories", "breakdown"]

    if any(word in text for word in add_words):
        return "To add an expense, use `!add <amount> <category> [note]`. Example: `!add 12.50 food lunch`."

    if any(word in text for word in list_words):
        return "To view your expenses, use `!list` for all entries or `!recent 5` for the latest ones."

    if any(word in text for word in recent_words):
        return "To see recent expenses, use `!recent [count]`. Example: `!recent 5`."

    if any(word in text for word in total_words):
        return "To see your total spending, use `!total`."

    if any(word in text for word in category_words):
        return "To see spending by category, use `!categories` or `!stats`."

    return "You can track expenses with `!add`, `!list`, `!recent`, `!total`, `!categories`, and `!stats`."


def get_rule_based_fallback(
    user_name: str,
    text: str,
    memory=None,
    recent_messages=None,
    conversation_summary: str = "",
) -> str:
    text = text.lower().strip()
    memory = memory or {}
    recent_messages = recent_messages or []

    remembered_name = memory.get("name", user_name)
    remembered_note = memory.get("note")
    remembered_preference = memory.get("preference")

    if is_greeting(text):
        return f"Hey {remembered_name}."

    if "who are you" in text or "what are you" in text:
        return "I'm Kiba Bot. I handle expense tracking and chat features."

    if (
        "what do you remember" in text
        or "what do you remember about me" in text
        or "what do you know about me" in text
        or "do you remember me" in text
        or "do you remember anything about me" in text
        or "what were we talking about earlier" in text
    ):
        parts = []

        if memory:
            parts.extend(f"{key}: {value}" for key, value in memory.items())

        if conversation_summary:
            parts.append(f"summary: {conversation_summary}")

        if not parts:
            return "I do not have anything remembered for you yet."

        return "Here is what I remember: " + "; ".join(parts)

    if is_thanks(text):
        return "You're welcome."

    if is_help_request(text):
        return get_help_response()

    if is_expense_topic(text):
        return get_expense_response(text)

    if "delete" in text or "remove" in text:
        return "To delete an expense, use `!delete <id>`. Example: `!delete 4`."

    if "clear" in text:
        return "To clear all expenses, use `!clear yes`."

    if "export" in text:
        return "To export your expenses, use `!export`."

    if "import" in text:
        return "To import expenses from a file, use `!import_expenses`."

    if "ping" in text:
        return "Use `!ping` to check bot latency."

    if is_private_info_request(text):
        return "I can't see your IP address, exact location, or device details through Discord chat."

    if remembered_note:
        return f"I remember this about you: {remembered_note}"

    if remembered_preference:
        return f"I remember your preference: {remembered_preference}"

    return "I heard you, but I do not know how to respond to that yet. Try `!helpchat` or `!help`."


_IMAGE_REQUEST = re.compile(
    r"(?:show|send|post|find|got|share)\s+(?:me\s+)?(?:a\s+|an\s+|any\s+|some\s+)?(.+?)(?:\s+with me)?$",
    re.IGNORECASE,
)

_MEDIA_KEYWORDS = re.compile(
    r"\b(?:meme|memes|gif|gifs|pic|pics|image|images|photo|photos|video|videos|guide|tutorial)\b",
    re.IGNORECASE,
)


def extract_image_request(text: str) -> str | None:
    """
    If the message is an explicit request to show/send/post media,
    return the topic keyword string. Otherwise return None.
    """
    m = _IMAGE_REQUEST.match(text.strip())
    if not m:
        return None
    topic = m.group(1).strip()
    # Must contain a media keyword to avoid false positives like "show me how to cook"
    if not _MEDIA_KEYWORDS.search(topic):
        return None
    return topic
