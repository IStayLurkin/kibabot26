import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from services.chat_service import ChatReply, generate_dynamic_reply


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_llm(reply="test response"):
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


def _patch_db():
    return [
        patch("services.chat_service.get_recent_chat_messages", new=AsyncMock(return_value=[])),
        patch("services.chat_service.get_user_memory", new=AsyncMock(return_value={})),
        patch("services.chat_service.get_conversation_summary", new=AsyncMock(return_value="")),
        patch("services.chat_service.get_conversation_state", new=AsyncMock(return_value={})),
        patch("services.chat_service.set_conversation_state", new=AsyncMock()),
    ]


class ChatReplyTests(unittest.TestCase):
    def test_chat_reply_is_namedtuple_like(self):
        reply = ChatReply(content="hello", intent="casual_chat")
        self.assertEqual(reply.content, "hello")
        self.assertEqual(reply.intent, "casual_chat")


class GenerateDynamicReplyTests(unittest.TestCase):
    def test_returns_chat_reply_instance(self):
        llm = _make_llm("hello back")
        patches = _patch_db()
        for p in patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(llm, "Brandon", "123", "456", 1, "hello"))
            self.assertIsInstance(result, ChatReply)
            self.assertIsInstance(result.content, str)
            self.assertTrue(len(result.content) > 0)
        finally:
            for p in patches:
                p.stop()

    def test_agentic_chat_disabled_by_default(self):
        """agentic_chat_enabled=False means generate_agent_reply is never called."""
        llm = _make_llm("direct reply")
        llm.agentic_chat_enabled = False
        patches = _patch_db()
        for p in patches:
            p.start()
        try:
            run(generate_dynamic_reply(llm, "Brandon", "123", "456", 1, "help me plan a project"))
            llm.generate_agent_reply.assert_not_called()
        finally:
            for p in patches:
                p.stop()

    def test_agentic_chat_enabled_calls_agent_reply_for_planning(self):
        llm = _make_llm("agent reply")
        llm.agentic_chat_enabled = True
        patches = _patch_db()
        for p in patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(llm, "Brandon", "123", "456", 1, "help me plan a project"))
            self.assertIsInstance(result, ChatReply)
            llm.generate_agent_reply.assert_called_once()
        finally:
            for p in patches:
                p.stop()

    def test_datetime_query_bypasses_llm(self):
        llm = _make_llm()
        patches = _patch_db()
        for p in patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(llm, "Brandon", "123", "456", 1, "what time is it"))
            self.assertIsInstance(result, ChatReply)
            llm.generate_reply.assert_not_called()
            llm.generate_agent_reply.assert_not_called()
        finally:
            for p in patches:
                p.stop()

    def test_behavior_rules_passed_to_llm(self):
        """Behavior rules loaded via behavior_rule_service are forwarded into generate_reply."""
        llm = _make_llm("reply")
        mock_rule_service = MagicMock()
        mock_rule_service.get_enabled_rule_texts = AsyncMock(return_value=["Do not use emojis."])
        mock_rule_service.add_rule = AsyncMock(return_value=(True, "Rule set."))
        mock_rule_service.edit_rule = AsyncMock(return_value=(True, "Rule updated."))
        mock_rule_service.get_rules_text = AsyncMock(return_value="No rules.")
        mock_rule_service.looks_like_rule_request = MagicMock(return_value=False)
        mock_rule_service.looks_like_rule_edit_request = MagicMock(return_value=False)
        patches = _patch_db()
        for p in patches:
            p.start()
        try:
            run(generate_dynamic_reply(
                llm, "Brandon", "123", "456", 1, "tell me something",
                services={"behavior_rule_service": mock_rule_service},
            ))
            call_kwargs = llm.generate_reply.call_args
            self.assertIsNotNone(call_kwargs)
            rules = call_kwargs.kwargs.get("behavior_rules")
            self.assertIsNotNone(rules)
            self.assertIn("Do not use emojis.", rules)
        finally:
            for p in patches:
                p.stop()

    def test_empty_message_returns_reply(self):
        llm = _make_llm("what?")
        patches = _patch_db()
        for p in patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(llm, "Brandon", "123", "456", 1, "   "))
            self.assertIsInstance(result, ChatReply)
        finally:
            for p in patches:
                p.stop()

    def test_memory_context_injected_from_db(self):
        """Memory rows from DB should be retrieved for context building."""
        llm = _make_llm("reply")
        # get_user_memory returns list of (key, value) tuples
        db_patches = [
            patch("services.chat_service.get_recent_chat_messages", new=AsyncMock(return_value=[])),
            patch("services.chat_service.get_user_memory", new=AsyncMock(return_value=[("name", "Brandon")])),
            patch("services.chat_service.get_conversation_summary", new=AsyncMock(return_value="")),
            patch("services.chat_service.get_conversation_state", new=AsyncMock(return_value={})),
            patch("services.chat_service.set_conversation_state", new=AsyncMock()),
        ]
        for p in db_patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(llm, "Brandon", "123", "456", 1, "hello"))
            self.assertIsInstance(result, ChatReply)
        finally:
            for p in db_patches:
                p.stop()

    def test_missing_llm_service_returns_fallback(self):
        patches = _patch_db()
        for p in patches:
            p.start()
        try:
            result = run(generate_dynamic_reply(None, "Brandon", "123", "456", 1, "hello"))
            self.assertIsInstance(result, ChatReply)
            self.assertIsInstance(result.content, str)
        finally:
            for p in patches:
                p.stop()


if __name__ == "__main__":
    unittest.main()
