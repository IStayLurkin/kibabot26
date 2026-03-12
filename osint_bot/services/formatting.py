from __future__ import annotations

from io import BytesIO

import discord

from osint_bot.core.config import OSINT_MAX_OUTPUT_CHARS
from osint_bot.core.constants import DISCORD_MESSAGE_SOFT_LIMIT
from osint_bot.services.models import OSINTResult


def render_result_text(result: OSINTResult) -> str:
    lines = [result.summary.strip()]

    if result.findings:
        lines.append("")
        lines.append("Findings:")
        lines.extend(f"- {line}" for line in result.findings)

    if result.sources:
        lines.append("")
        lines.append("Sources:")
        lines.extend(f"- {source}" for source in result.sources)

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)

    if result.blocked_reason:
        lines.append("")
        lines.append(f"Blocked: {result.blocked_reason}")

    return "\n".join(lines).strip()


def build_discord_payload(result: OSINTResult) -> tuple[str, discord.File | None]:
    text = render_result_text(result)
    max_chars = min(OSINT_MAX_OUTPUT_CHARS, DISCORD_MESSAGE_SOFT_LIMIT)

    if len(text) <= max_chars:
        return text, None

    summary = result.summary.strip()[: max_chars - 64].rstrip()
    message = f"{summary}\n\nFull results attached."
    file_obj = BytesIO(text.encode("utf-8"))
    attachment = discord.File(file_obj, filename="osint_result.txt")
    return message, attachment
