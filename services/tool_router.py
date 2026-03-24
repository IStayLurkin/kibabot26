from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


INTENT_CASUAL_CHAT = "casual_chat"
INTENT_QUESTION_ANSWERING = "question_answering"
INTENT_MULTI_STEP_HELP = "multi_step_help"
INTENT_PLANNING = "planning"
INTENT_TROUBLESHOOTING = "troubleshooting"
INTENT_TOOL_USE_REQUEST = "tool_use_request"
INTENT_CODE_GENERATION_ANALYSIS = "code_generation_analysis"


@dataclass(slots=True)
class RouteDecision:
    intent: str
    confidence: float
    requires_agent: bool
    should_ask_clarifying_question: bool
    tool_name: str = ""
    tool_input: str = ""


class ToolRouter:
    NON_MEDIA_HINTS = (
        "rule",
        "policy",
        "reply",
        "response",
        "message",
        "command",
        "memory",
        "remember",
        "emoji",
        "emojis",
    )

    IMAGE_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"^(?:!image|!img)\b",
        r"\b(?:generate|make|create|draw)\b.*\b(?:image|picture|art|photo)\b",
        r"^(?:draw me|draw|generate (?:me )?(?:a|an)?|create (?:me )?(?:a|an)?|make (?:me )?(?:a|an)?)\b",
        r"^(?:color|colour)\s+(?:a\s+)?(?:pic|picture|photo|image)\s+of\b",
    )]

    VOICE_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"^(?:!tts|!say)\b",
        r"\b(?:text to speech|tts|voice gen|generate voice|say this)\b",
    )]

    VIDEO_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"^(?:!video|!animate)\b",
        r"\b(?:generate|make|create)\b.*\bvideo\b",
    )]

    MUSIC_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"^(?:!melody|!music|!tune)\b",
        r"\b(?:generate|make|create|compose)\b.*\b(?:melody|music|tune|beat|loop)\b",
        r"^(?:compose|make|create)\s+(?:me\s+)?(?:a\s+)?(?:melody|tune|beat|loop)\b",
    )]

    CODE_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"^(?:!code|!fixcode|!explaincode|!refactor)\b",
        r"\b(?:write code|fix code|debug code|refactor code|explain code|review this code|analyze this code)\b",
    )]

    OSINT_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"^(?:!osint|!whois|!domain)\b",
        r"\b(?:whois|rdap|dns|domain lookup|public intel|osint|look up this domain)\b",
    )]

    PLANNING_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"\b(?:plan|roadmap|steps|step by step|walk me through|organize this)\b",
    )]

    TROUBLESHOOTING_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"\b(?:error|bug|broken|not working|failed|crash|issue|problem|traceback|fix this)\b",
    )]

    MULTI_STEP_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"\b(?:help me|show me how|how do i|how should i|what should i do|best way to)\b",
    )]

    CASUAL_PATTERNS: ClassVar[list] = [re.compile(p) for p in (
        r"^(?:hi|hello|hey|yo|sup)\b",
        r"\b(?:how are you|what's up|wyd)\b",
    )]

    def route(self, content: str) -> RouteDecision:
        text = content.strip()
        lowered = text.lower()

        tool_name = self.detect_tool(lowered)
        if tool_name:
            tool_input = self.extract_tool_input(text, tool_name)
            return RouteDecision(
                intent=(
                    INTENT_CODE_GENERATION_ANALYSIS
                    if tool_name == "code"
                    else INTENT_TOOL_USE_REQUEST
                ),
                confidence=0.95,
                requires_agent=True,
                should_ask_clarifying_question=self._tool_needs_more_input(tool_name, tool_input),
                tool_name=tool_name,
                tool_input=tool_input,
            )

        if self._matches_any(lowered, self.TROUBLESHOOTING_PATTERNS):
            return RouteDecision(
                intent=INTENT_TROUBLESHOOTING,
                confidence=0.9,
                requires_agent=True,
                should_ask_clarifying_question=self._needs_clarifying_question(lowered),
            )

        if self._matches_any(lowered, self.PLANNING_PATTERNS):
            return RouteDecision(
                intent=INTENT_PLANNING,
                confidence=0.88,
                requires_agent=True,
                should_ask_clarifying_question=self._needs_clarifying_question(lowered),
            )

        if self._matches_any(lowered, self.MULTI_STEP_PATTERNS):
            return RouteDecision(
                intent=INTENT_MULTI_STEP_HELP,
                confidence=0.82,
                requires_agent=True,
                should_ask_clarifying_question=self._needs_clarifying_question(lowered),
            )

        if self._looks_like_code_request(lowered):
            return RouteDecision(
                intent=INTENT_CODE_GENERATION_ANALYSIS,
                confidence=0.88,
                requires_agent=True,
                should_ask_clarifying_question=self._needs_clarifying_question(lowered),
            )

        if self._matches_any(lowered, self.CASUAL_PATTERNS) and len(lowered.split()) <= 8:
            return RouteDecision(
                intent=INTENT_CASUAL_CHAT,
                confidence=0.78,
                requires_agent=False,
                should_ask_clarifying_question=False,
            )

        return RouteDecision(
            intent=INTENT_QUESTION_ANSWERING,
            confidence=0.65,
            requires_agent=False,
            should_ask_clarifying_question=False,
        )

    def detect_tool(self, text: str) -> str:
        if not self._looks_non_media_request(text):
            for pattern in self.IMAGE_PATTERNS:
                if pattern.search(text):
                    return "image"

        for pattern in self.VOICE_PATTERNS:
            if pattern.search(text):
                return "voice"

        for pattern in self.VIDEO_PATTERNS:
            if pattern.search(text):
                return "video"

        for pattern in self.MUSIC_PATTERNS:
            if pattern.search(text):
                return "music"

        for pattern in self.CODE_PATTERNS:
            if pattern.search(text):
                return "code"

        for pattern in self.OSINT_PATTERNS:
            if pattern.search(text):
                return "osint"

        return ""

    def extract_tool_input(self, text: str, tool_name: str) -> str:
        cleaned = text.strip()
        lowered = cleaned.lower()

        prefix_patterns = {
            "image": [r"^!image\s+", r"^!img\s+"],
            "voice": [r"^!tts\s+", r"^!say\s+"],
            "video": [r"^!video\s+", r"^!animate\s+"],
            "music": [r"^!melody\s+", r"^!music\s+", r"^!tune\s+"],
            "code": [r"^!code\s+", r"^!fixcode\s+", r"^!explaincode\s+", r"^!refactor\s+"],
            "osint": [r"^!osint\s+", r"^!whois\s+", r"^!domain\s+"],
        }

        for pattern in prefix_patterns.get(tool_name, []):
            if re.search(pattern, lowered):
                return re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        if tool_name == "voice":
            match = re.search(r"(?:say this|tts|text to speech)\s*[:\-]?\s*(.+)", cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if tool_name == "image":
            image_patterns = [
                r"^(?:draw me)\s+",
                r"^(?:draw)\s+",
                r"^(?:generate (?:me )?(?:a|an)?)\s+",
                r"^(?:create (?:me )?(?:a|an)?)\s+",
                r"^(?:make (?:me )?(?:a|an)?)\s+",
                r"^(?:color|colour)\s+(?:a\s+)?(?:pic|picture|photo|image)\s+of\s+",
            ]

            for pattern in image_patterns:
                if re.search(pattern, lowered, flags=re.IGNORECASE):
                    extracted = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" .,:;-")
                    if extracted:
                        return extracted

        if tool_name == "music":
            music_patterns = [
                r"^(?:compose)\s+(?:me\s+)?(?:a\s+)?(?:melody|tune|beat|loop)\s*",
                r"^(?:make|create|generate)\s+(?:me\s+)?(?:a\s+)?(?:melody|tune|beat|loop)\s*",
            ]

            for pattern in music_patterns:
                if re.search(pattern, lowered, flags=re.IGNORECASE):
                    extracted = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" .,:;-")
                    if extracted:
                        return extracted

        if tool_name == "osint":
            match = re.search(r"(?:whois|rdap|dns|domain lookup|look up this domain)\s+(.+)", cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return cleaned

    def _tool_needs_more_input(self, tool_name: str, tool_input: str) -> bool:
        if tool_name in {"image", "video", "voice", "music", "code", "osint"}:
            return len(tool_input.strip()) < 3
        return False

    def _needs_clarifying_question(self, text: str) -> bool:
        vague_requests = (
            "help me",
            "fix this",
            "it broke",
            "it failed",
            "plan this",
            "what should i do",
        )
        if any(phrase == text for phrase in vague_requests):
            return True

        return len(text.split()) <= 2

    def _looks_like_code_request(self, text: str) -> bool:
        code_markers = ("python", "javascript", "typescript", "function(", "def ", "script", "stack trace", "traceback")
        return any(marker in text for marker in code_markers)

    def _matches_any(self, text: str, patterns: list) -> bool:
        return any(p.search(text) for p in patterns)

    def _looks_non_media_request(self, text: str) -> bool:
        lowered = text.strip().lower()
        if not lowered.startswith(("create ", "make ", "generate ")):
            return False
        return any(hint in lowered for hint in self.NON_MEDIA_HINTS)
