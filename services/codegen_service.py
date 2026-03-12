from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ToolIntent:
    name: str
    confidence: float


class ToolRouter:
    IMAGE_PATTERNS = [
        r"^(?:!image|!img)\b",
        r"\b(?:generate|make|create|draw)\b.*\b(?:image|picture|art|photo)\b",
    ]

    VOICE_PATTERNS = [
        r"^(?:!tts|!say)\b",
        r"\b(?:text to speech|tts|voice gen|generate voice)\b",
    ]

    VIDEO_PATTERNS = [
        r"^(?:!video|!animate)\b",
        r"\b(?:generate|make|create)\b.*\bvideo\b",
    ]

    CODE_PATTERNS = [
        r"^(?:!code|!fixcode|!explaincode|!refactor)\b",
        r"\b(?:write code|fix code|debug code|refactor code|explain code)\b",
    ]

    OSINT_PATTERNS = [
        r"^(?:!osint|!whois|!domain|!username)\b",
        r"\b(?:whois|rdap|dns|domain lookup|public intel|osint)\b",
    ]

    def detect_intent(self, content: str) -> ToolIntent:
        text = content.strip().lower()

        for pattern in self.IMAGE_PATTERNS:
            if re.search(pattern, text):
                return ToolIntent("image", 0.95)

        for pattern in self.VOICE_PATTERNS:
            if re.search(pattern, text):
                return ToolIntent("voice", 0.95)

        for pattern in self.VIDEO_PATTERNS:
            if re.search(pattern, text):
                return ToolIntent("video", 0.95)

        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, text):
                return ToolIntent("code", 0.95)

        for pattern in self.OSINT_PATTERNS:
            if re.search(pattern, text):
                return ToolIntent("osint", 0.95)

        return ToolIntent("chat", 0.25)