import pytest
from unittest.mock import MagicMock, patch
from services.llm_service import LLMService


@pytest.fixture
def llm():
    return LLMService()


def test_extract_episodic_returns_content_when_worthy(llm):
    with patch.object(llm, "_complete_messages_sync", return_value='{"should_store": true, "content": "Brandon is building a Discord bot"}'):
        result = llm.extract_episodic_memory_sync("I am building a Discord bot", "Cool!")
    assert result["should_store"] is True
    assert result["content"] == "Brandon is building a Discord bot"


def test_extract_episodic_returns_false_for_casual(llm):
    with patch.object(llm, "_complete_messages_sync", return_value='{"should_store": false, "content": ""}'):
        result = llm.extract_episodic_memory_sync("hey", "hey!")
    assert result["should_store"] is False


def test_extract_episodic_returns_false_on_malformed_json(llm):
    with patch.object(llm, "_complete_messages_sync", return_value="not json"):
        result = llm.extract_episodic_memory_sync("some message", "some reply")
    assert result["should_store"] is False


def test_extract_episodic_returns_false_on_exception(llm):
    with patch.object(llm, "_complete_messages_sync", side_effect=RuntimeError("LLM down")):
        result = llm.extract_episodic_memory_sync("some message", "some reply")
    assert result["should_store"] is False
