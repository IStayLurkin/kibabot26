import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE = "America/Los_Angeles"


def get_now(timezone_name: str = DEFAULT_TIMEZONE):
    try:
        tz = ZoneInfo(timezone_name)
        return datetime.now(tz), timezone_name
    except ZoneInfoNotFoundError:
        now = datetime.now().astimezone()
        return now, str(now.tzinfo) if now.tzinfo else "local time"


def format_current_datetime_context(timezone_name: str = DEFAULT_TIMEZONE) -> str:
    now, tz_name = get_now(timezone_name)

    return (
        f"Current date: {now.strftime('%Y-%m-%d')}\n"
        f"Current day: {now.strftime('%A')}\n"
        f"Current local time: {now.strftime('%I:%M %p %Z')}\n"
        f"Current timezone: {tz_name}\n"
        f"Current full datetime: {now.strftime('%A, %B %d, %Y at %I:%M %p %Z')}"
    )


def is_date_time_question(text: str) -> bool:
    lowered = text.lower().strip()

    patterns = [
        r"\bwhat day is (it|today)\b",
        r"\bwhat day today\b",
        r"\bwhat'?s the date\b",
        r"\bwhat is the date\b",
        r"\btoday'?s date\b",
        r"\bwhat time is it\b",
        r"\bcurrent time\b",
        r"\bcurrent date\b",
        r"\bwhat year is it\b",
        r"\bwhat month is it\b",
        r"\bwhat day of the week is it\b",
        r"\bwhat day of the week is today\b",
    ]

    return any(re.search(pattern, lowered) for pattern in patterns)


def build_current_datetime_reply(text: str, timezone_name: str = DEFAULT_TIMEZONE) -> str:
    lowered = text.lower().strip()
    now, _tz_name = get_now(timezone_name)

    if "what time is it" in lowered or "current time" in lowered:
        return f"It is {now.strftime('%I:%M %p %Z')} on {now.strftime('%A, %B %d, %Y')}."

    if "what year is it" in lowered:
        return f"It is {now.strftime('%Y')}."

    if "what month is it" in lowered:
        return f"It is {now.strftime('%B %Y')}."

    if (
        "what day is today" in lowered
        or "what day is it" in lowered
        or "what day today" in lowered
        or "what day of the week is it" in lowered
        or "what day of the week is today" in lowered
    ):
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    if (
        "what's the date" in lowered
        or "what is the date" in lowered
        or "today's date" in lowered
        or "current date" in lowered
    ):
        return f"Today's date is {now.strftime('%B %d, %Y')}."

    return f"It is {now.strftime('%A, %B %d, %Y at %I:%M %p %Z')}."
