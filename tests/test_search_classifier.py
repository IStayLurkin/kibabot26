import pytest
from unittest.mock import MagicMock, patch
from services.llm_service import LLMService


@pytest.fixture
def llm():
    return LLMService()


def test_classifier_returns_queries_for_current_events(llm):
    with patch.object(llm, "_complete_messages_sync", return_value='["2026 NBA finals winner", "NBA championship 2026"]'):
        queries = llm._classify_search_need("who won the NBA finals this year?")
    assert queries == ["2026 NBA finals winner", "NBA championship 2026"]


def test_classifier_returns_empty_for_casual_chat(llm):
    with patch.object(llm, "_complete_messages_sync", return_value="null"):
        queries = llm._classify_search_need("hey what's up")
    assert queries == []


def test_classifier_returns_empty_on_malformed_json(llm):
    with patch.object(llm, "_complete_messages_sync", return_value="not json at all"):
        queries = llm._classify_search_need("some message")
    assert queries == []


def test_classifier_returns_empty_on_llm_exception(llm):
    with patch.object(llm, "_complete_messages_sync", side_effect=RuntimeError("LLM down")):
        queries = llm._classify_search_need("some message")
    assert queries == []


def test_classifier_caps_at_three_queries(llm):
    with patch.object(llm, "_complete_messages_sync", return_value='["q1", "q2", "q3", "q4", "q5"]'):
        queries = llm._classify_search_need("some complex question")
    assert len(queries) == 3
