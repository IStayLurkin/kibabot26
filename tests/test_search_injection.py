import pytest
from services.llm_service import LLMService


@pytest.fixture
def llm():
    return LLMService()


def test_build_messages_injects_search_results(llm):
    results = [
        {"title": "AI News 2026", "snippet": "Big AI developments.", "url": "http://example.com/ai"},
    ]
    messages = llm._build_messages(
        user_display_name="Brandon",
        user_message="what's new in AI?",
        memory={},
        recent_messages=[],
        search_results=results,
    )
    system_content = messages[0]["content"]
    assert "SEARCH RESULTS" in system_content
    assert "AI News 2026" in system_content
    assert "Big AI developments." in system_content


def test_build_messages_no_search_results_no_block(llm):
    messages = llm._build_messages(
        user_display_name="Brandon",
        user_message="hey",
        memory={},
        recent_messages=[],
        search_results=[],
    )
    system_content = messages[0]["content"]
    assert "SEARCH RESULTS" not in system_content


def test_build_messages_search_results_default_none(llm):
    # Existing callers that don't pass search_results should still work
    messages = llm._build_messages(
        user_display_name="Brandon",
        user_message="hey",
        memory={},
        recent_messages=[],
    )
    system_content = messages[0]["content"]
    assert "SEARCH RESULTS" not in system_content
