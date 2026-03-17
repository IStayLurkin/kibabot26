from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from core.feature_flags import AGENT_DEFAULT_COOLDOWN_SECONDS

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self, llm_service: Any | None = None) -> None:
        self.llm_service = llm_service
        self.enabled_channels: dict[int, set[int]] = defaultdict(set)
        self.last_action_at: dict[tuple[int, int], float] = {}

    def enable_channel(self, guild_id: int, channel_id: int) -> None:
        self.enabled_channels[guild_id].add(channel_id)
        logger.info("Agent enabled for guild=%s channel=%s", guild_id, channel_id)

    def disable_channel(self, guild_id: int, channel_id: int) -> None:
        if guild_id in self.enabled_channels:
            self.enabled_channels[guild_id].discard(channel_id)
        logger.info("Agent disabled for guild=%s channel=%s", guild_id, channel_id)

    def is_enabled(self, guild_id: int, channel_id: int) -> bool:
        return channel_id in self.enabled_channels.get(guild_id, set())

    def get_status(self, guild_id: int, channel_id: int) -> str:
        return "enabled" if self.is_enabled(guild_id, channel_id) else "disabled"

    async def maybe_handle_game_message(self, message: Any) -> bool:
        """
        Safe starter behavior:
        - only in explicitly enabled channels
        - cooldown protected
        - only reacts to obvious turn prompts / game prompts
        """
        guild = getattr(message, "guild", None)
        channel = getattr(message, "channel", None)

        if guild is None or channel is None:
            return False

        if not self.is_enabled(guild.id, channel.id):
            return False

        if getattr(message.author, "bot", False):
            return False

        content = (message.content or "").strip()
        if not content:
            return False

        if not self._looks_like_game_prompt(content):
            return False

        if not self._cooldown_ok(guild.id, channel.id):
            return False

        response = await self._decide_response(content)
        if not response:
            return False

        await asyncio.sleep(1.0)
        await message.channel.send(response)
        self.last_action_at[(guild.id, channel.id)] = time.time()
        return True

    def _looks_like_game_prompt(self, content: str) -> bool:
        text = content.lower()
        keywords = (
            "your turn",
            "choose",
            "pick one",
            "roll",
            "attack",
            "defend",
            "move",
            "guess",
            "question:",
            "trivia",
            "battle",
            "play card",
        )
        return any(keyword in text for keyword in keywords)

    def _cooldown_ok(self, guild_id: int, channel_id: int) -> bool:
        key = (guild_id, channel_id)
        last = self.last_action_at.get(key, 0.0)
        return (time.time() - last) >= AGENT_DEFAULT_COOLDOWN_SECONDS

    async def _decide_response(self, content: str) -> str:
        if self.llm_service is None:
            return self._simple_rule_response(content)

        prompt = (
            "You are a careful Discord game assistant.\n"
            "Reply with exactly one short move/message suitable for the game prompt.\n"
            "Do not explain your reasoning.\n"
            "Do not roleplay outside the game.\n\n"
            f"Game prompt:\n{content}"
        )

        for method_name in ("generate_response", "generate_text", "get_response", "complete", "chat"):
            method = getattr(self.llm_service, method_name, None)
            if method is None:
                continue
            try:
                result = await method(prompt)
                return str(result).strip()[:300]
            except Exception:
                logger.exception("Agent LLM response failed via %s", method_name)

        return self._simple_rule_response(content)

    def _simple_rule_response(self, content: str) -> str:
        text = content.lower()

        if "roll" in text:
            return "roll"
        if "choose" in text or "pick one" in text:
            return "1"
        if "attack" in text:
            return "attack"
        if "defend" in text:
            return "defend"
        if "guess" in text:
            return "A"

        return "pass"