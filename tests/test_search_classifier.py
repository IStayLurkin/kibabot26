import pytest
from unittest.mock import MagicMock, patch
from services.llm_service import LLMService, _message_needs_search


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


# --- _message_needs_search pre-filter tests ---

def test_needs_search_current_events():
    assert _message_needs_search("who won the super bowl this year?") is True

def test_needs_search_latest_news():
    assert _message_needs_search("what's the latest news on AI?") is True

def test_needs_search_recent():
    assert _message_needs_search("any recent updates on GPT?") is True

def test_needs_search_today():
    assert _message_needs_search("what happened today in politics?") is True

def test_needs_search_price():
    assert _message_needs_search("what's the price of bitcoin right now?") is True

def test_needs_search_score():
    assert _message_needs_search("what's the score of the lakers game?") is True

def test_no_search_casual():
    assert _message_needs_search("hey what's up") is False

def test_no_search_greeting():
    assert _message_needs_search("how are you doing") is False

def test_no_search_personal():
    assert _message_needs_search("tell me a joke") is False

def test_no_search_math():
    assert _message_needs_search("what is 2 plus 2") is False
