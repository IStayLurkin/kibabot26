import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from services.chat_service import ChatReply, generate_dynamic_reply


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class ChatRoutingTests(unittest.TestCase):
    def _make_llm(self, reply="test response"):
        llm = MagicMock()
        llm.generate_reply = AsyncMock(return_value=reply)
        llm.generate_agent_reply = AsyncMock(return_value={
            "intent": "casual_chat",
            "goal": "",
            "response_mode": "direct",
            "needs_clarification": False,
            "clarifying_question": "",
            "tool_suggestion": "",
            "tool_reason": "",
            "answer": reply,
            "next_steps": [],
            "state_update": {"goal": "", "pending_question": ""},
        })
        llm.agentic_chat_enabled = False
        llm.performance_tracker = None
        llm.timezone_name = "America/Los_Angeles"
        return llm

    def _patch_db(self):
        """Patch all database calls to return empty/safe defaults."""
        patches = [
            patch("services.chat_service.get_recent_chat_messages", new=AsyncMock(return_value=[])),
            patch("services.chat_service.get_user_memory", new=AsyncMock(return_value={})),
            patch("services.chat_service.get_conversation_summary", new=AsyncMock(return_value="")),
            patch("services.chat_service.get_conversation_state", new=AsyncMock(return_value={})),
            patch("services.chat_service.set_conversation_state", new=AsyncMock()),
        ]
        return patches

    def test_returns_chat_reply(self):
        llm = self._make_llm("hello back")
        patches = self._patch_db()
        for p in patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(
                llm, "Brandon", "123", "456", 1, "hello"
            ))
            self.assertIsInstance(result, ChatReply)
            self.assertIsInstance(result.content, str)
            self.assertTrue(len(result.content) > 0)
        finally:
            for p in patches:
                p.stop()

    def test_datetime_question_bypasses_llm(self):
        llm = self._make_llm()
        patches = self._patch_db()
        for p in patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(
                llm, "Brandon", "123", "456", 1, "what time is it"
            ))
            self.assertIsInstance(result, ChatReply)
            # LLM should not have been called for a time question
            llm.generate_reply.assert_not_called()
        finally:
            for p in patches:
                p.stop()

    def test_missing_osint_service_returns_graceful_message(self):
        from services.chat_service import _run_tool
        result = run(_run_tool("osint", "example.com", {}))
        self.assertIsInstance(result, ChatReply)
        self.assertIn("not available", result.content.lower())

    def test_missing_image_service_returns_graceful_message(self):
        from services.chat_service import _run_tool
        result = run(_run_tool("image", "a cat", {}))
        self.assertIsInstance(result, ChatReply)
        self.assertIn("not available", result.content.lower())

    def test_unknown_tool_returns_none(self):
        from services.chat_service import _run_tool
        result = run(_run_tool("nonexistent_tool", "input", {}))
        self.assertIsNone(result)

    def test_runtime_service_query_bypasses_llm(self):
        llm = self._make_llm()
        mock_runtime = MagicMock()
        mock_runtime.answer_natural_language_query.return_value = "Active provider: ollama"
        patches = self._patch_db()
        for p in patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(
                llm, "Brandon", "123", "456", 1,
                "what model are you using",
                services={"model_runtime_service": mock_runtime},
            ))
            self.assertIsInstance(result, ChatReply)
            self.assertIn("ollama", result.content)
            llm.generate_reply.assert_not_called()
        finally:
            for p in patches:
                p.stop()


if __name__ == "__main__":
    unittest.main()
