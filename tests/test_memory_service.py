import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from services.memory_service import (
    extract_memory_fact,
    maybe_extract_ai_memory,
    should_attempt_memory_storage,
    store_memory_if_found,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class ExtractMemoryFactTests(unittest.TestCase):
    def test_my_name_is(self):
        key, value = extract_memory_fact("My name is Brandon")
        self.assertEqual(key, "name")
        self.assertIn("Brandon", value)

    def test_remember_that(self):
        key, value = extract_memory_fact("remember that I like dark mode")
        self.assertEqual(key, "note")
        self.assertIn("dark mode", value)

    def test_remember_prefix(self):
        key, value = extract_memory_fact("remember I prefer Python over JS")
        self.assertEqual(key, "note")

    def test_i_prefer(self):
        key, value = extract_memory_fact("I prefer no emojis")
        self.assertEqual(key, "preference")
        self.assertIn("no emojis", value)

    def test_stop_emojis(self):
        key, value = extract_memory_fact("stop sending emojis")
        self.assertEqual(key, "emoji_preference")

    def test_do_not_use_emojis(self):
        key, value = extract_memory_fact("do not use emojis unless requested")
        self.assertEqual(key, "emoji_preference")

    def test_no_match_returns_none(self):
        result = extract_memory_fact("what time is it")
        self.assertIsNone(result)

    def test_empty_returns_none(self):
        result = extract_memory_fact("")
        self.assertIsNone(result)


class ShouldAttemptMemoryStorageTests(unittest.TestCase):
    def test_explicit_remember_returns_true(self):
        self.assertTrue(should_attempt_memory_storage("remember that I like Python"))

    def test_non_memory_prefix_blocked(self):
        self.assertFalse(should_attempt_memory_storage("help me debug this"))
        self.assertFalse(should_attempt_memory_storage("generate an image of a cat"))
        self.assertFalse(should_attempt_memory_storage("how do i fix this"))

    def test_blocked_phrases_blocked(self):
        self.assertFalse(should_attempt_memory_storage("my budget is tight"))
        self.assertFalse(should_attempt_memory_storage("there is a bug in my code"))

    def test_too_short_blocked(self):
        self.assertFalse(should_attempt_memory_storage("hi"))
        self.assertFalse(should_attempt_memory_storage(""))

    def test_casual_chat_blocked(self):
        self.assertFalse(should_attempt_memory_storage("hey"))


class MaybeExtractAiMemoryTests(unittest.TestCase):
    def _make_llm(self, should_store=True, key="theme", value="dark mode"):
        llm = MagicMock()
        llm.extract_memory = AsyncMock(return_value={
            "should_store": should_store,
            "memory_key": key,
            "memory_value": value,
        })
        llm.behavior_rule_service = None
        return llm

    def test_stores_valid_memory(self):
        llm = self._make_llm(should_store=True, key="theme", value="dark mode")
        with patch("services.memory_service.get_user_memory", new=AsyncMock(return_value=[])), \
             patch("services.memory_service.set_user_memory", new=AsyncMock()) as mock_set:
            result = run(maybe_extract_ai_memory(llm, "123", "I prefer dark mode"))
            self.assertIsNotNone(result)
            self.assertEqual(result[0], "theme")
            mock_set.assert_called_once()

    def test_skips_when_should_store_false(self):
        llm = self._make_llm(should_store=False)
        with patch("services.memory_service.get_user_memory", new=AsyncMock(return_value=[])), \
             patch("services.memory_service.set_user_memory", new=AsyncMock()) as mock_set:
            result = run(maybe_extract_ai_memory(llm, "123", "I prefer dark mode"))
            self.assertIsNone(result)
            mock_set.assert_not_called()

    def test_blocks_finance_keys(self):
        llm = self._make_llm(should_store=True, key="budget", value="tight")
        with patch("services.memory_service.get_user_memory", new=AsyncMock(return_value=[])), \
             patch("services.memory_service.set_user_memory", new=AsyncMock()) as mock_set:
            result = run(maybe_extract_ai_memory(llm, "123", "my budget is tight"))
            self.assertIsNone(result)
            mock_set.assert_not_called()

    def test_blocks_blocked_finance_keys_income(self):
        llm = self._make_llm(should_store=True, key="income", value="5000")
        with patch("services.memory_service.get_user_memory", new=AsyncMock(return_value=[])), \
             patch("services.memory_service.set_user_memory", new=AsyncMock()) as mock_set:
            result = run(maybe_extract_ai_memory(llm, "123", "my income is 5000"))
            self.assertIsNone(result)
            mock_set.assert_not_called()

    def test_blocks_value_over_word_limit(self):
        long_value = " ".join(["word"] * 25)
        llm = self._make_llm(should_store=True, key="note", value=long_value)
        with patch("services.memory_service.get_user_memory", new=AsyncMock(return_value=[])), \
             patch("services.memory_service.set_user_memory", new=AsyncMock()) as mock_set:
            result = run(maybe_extract_ai_memory(llm, "123", "remember I prefer this"))
            self.assertIsNone(result)
            mock_set.assert_not_called()

    def test_value_within_word_limit_stores(self):
        llm = self._make_llm(should_store=True, key="note", value="dark mode always")
        with patch("services.memory_service.get_user_memory", new=AsyncMock(return_value=[])), \
             patch("services.memory_service.set_user_memory", new=AsyncMock()) as mock_set:
            result = run(maybe_extract_ai_memory(llm, "123", "remember I prefer dark mode always"))
            self.assertIsNotNone(result)
            mock_set.assert_called_once()

    def test_non_memory_content_skipped_before_llm(self):
        llm = self._make_llm()
        with patch("services.memory_service.get_user_memory", new=AsyncMock(return_value=[])), \
             patch("services.memory_service.set_user_memory", new=AsyncMock()) as mock_set:
            result = run(maybe_extract_ai_memory(llm, "123", "help me fix this bug"))
            self.assertIsNone(result)
            mock_set.assert_not_called()


class StoreMemoryIfFoundTests(unittest.TestCase):
    def test_explicit_memory_stored_directly(self):
        with patch("services.memory_service.set_user_memory", new=AsyncMock()) as mock_set:
            result = run(store_memory_if_found(MagicMock(), "123", "my name is Brandon"))
            self.assertIsNotNone(result)
            self.assertEqual(result[0], "name")
            mock_set.assert_called_once()

    def test_non_explicit_falls_through_to_ai(self):
        llm = MagicMock()
        llm.extract_memory = AsyncMock(return_value={"should_store": False})
        llm.behavior_rule_service = None
        with patch("services.memory_service.get_user_memory", new=AsyncMock(return_value=[])), \
             patch("services.memory_service.set_user_memory", new=AsyncMock()):
            result = run(store_memory_if_found(llm, "123", "what's the weather"))
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
