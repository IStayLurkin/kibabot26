# Web Search / RAG via SearXNG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add real-time web search to the Discord bot so the LLM can answer questions about current events by querying SearXNG and injecting results into context.

**Architecture:** An LLM classifier call determines whether the user's message needs real-world information; if yes, it returns 1-3 search queries that `SearchService` runs in parallel against SearXNG. Results are injected as a `[SEARCH RESULTS]` block into `_build_messages` alongside the existing preamble, then the normal response flow continues. Failures are silent — if SearXNG is down the bot answers without search context.

**Tech Stack:** SearXNG (Docker), `aiohttp` for async HTTP, existing Ollama LLM for classifier, `asyncio.gather` for parallel queries.

---

### Task 1: Start SearXNG in Docker

**Files:**
- No code changes — this is a one-time setup step

**Step 1: Run SearXNG container**

```bash
docker run -d --name searxng -p 8080:8080 --restart unless-stopped searxng/searxng
```

**Step 2: Verify it's running**

```bash
curl "http://localhost:8080/search?q=test&format=json"
```

Expected: JSON response with a `results` array (may be empty for "test" but the structure should be there).

**Step 3: Commit nothing** — this is infra setup only.

---

### Task 2: Add SearXNG config to `core/config.py`

**Files:**
- Modify: `core/config.py` (end of file)
- Modify: `.env` (add new vars)

**Step 1: Add config vars to `core/config.py`**

Append to the end of `core/config.py`:

```python
SEARXNG_ENABLED = os.getenv("SEARXNG_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080").strip()
SEARXNG_MAX_RESULTS = _parse_int(os.getenv("SEARXNG_MAX_RESULTS", "3"), 3)
```

**Step 2: Add defaults to `.env`**

Add these lines to `.env`:

```
SEARXNG_ENABLED=true
SEARXNG_BASE_URL=http://localhost:8080
SEARXNG_MAX_RESULTS=3
```

**Step 3: Commit**

```bash
git add core/config.py .env
git commit -m "feat: add SearXNG config vars"
```

---

### Task 3: Create `SearchService` with failing tests first

**Files:**
- Create: `services/search_service.py`
- Create: `tests/test_search_service.py`

**Step 1: Write the failing tests**

Create `tests/test_search_service.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.search_service import SearchService


@pytest.fixture
def service():
    return SearchService(base_url="http://localhost:8080", max_results=3)


@pytest.mark.asyncio
async def test_search_returns_formatted_results(service):
    fake_response = {
        "results": [
            {"title": "Result One", "content": "Snippet one.", "url": "http://example.com/1"},
            {"title": "Result Two", "content": "Snippet two.", "url": "http://example.com/2"},
        ]
    }
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        results = await service.search("latest AI news")

    assert len(results) == 2
    assert results[0]["title"] == "Result One"
    assert results[0]["snippet"] == "Snippet one."
    assert results[0]["url"] == "http://example.com/1"


@pytest.mark.asyncio
async def test_search_respects_max_results(service):
    fake_response = {
        "results": [
            {"title": f"Result {i}", "content": f"Snippet {i}.", "url": f"http://example.com/{i}"}
            for i in range(10)
        ]
    }
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        results = await service.search("query")

    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_returns_empty_on_http_error(service):
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        results = await service.search("query")

    assert results == []


@pytest.mark.asyncio
async def test_search_returns_empty_on_network_error(service):
    import aiohttp
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError("connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        results = await service.search("query")

    assert results == []


@pytest.mark.asyncio
async def test_search_multiple_queries_parallel(service):
    fake_response = {
        "results": [
            {"title": "Result", "content": "Snippet.", "url": "http://example.com/1"},
        ]
    }
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=fake_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        all_results = await service.search_many(["query one", "query two"])

    # Each query returns 1 result, 2 queries total
    assert len(all_results) == 2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_search_service.py -v --timeout=5
```

Expected: `ModuleNotFoundError: No module named 'services.search_service'`

**Step 3: Install `aiohttp` if not already present**

```bash
pip install aiohttp
```

**Step 4: Create `services/search_service.py`**

```python
import asyncio
import logging
import urllib.parse

import aiohttp

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT_SECONDS = 5


class SearchService:
    def __init__(self, base_url: str, max_results: int = 3):
        self._base_url = base_url.rstrip("/")
        self._max_results = max_results

    async def search(self, query: str) -> list[dict]:
        """Search SearXNG for query. Returns list of {title, snippet, url} dicts. Never raises."""
        url = f"{self._base_url}/search"
        params = {"q": query, "format": "json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=_SEARCH_TIMEOUT_SECONDS),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("SearXNG returned status %d for query %r", resp.status, query)
                        return []
                    data = await resp.json()
                    results = data.get("results", [])
                    return [
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("content", ""),
                            "url": r.get("url", ""),
                        }
                        for r in results[: self._max_results]
                        if r.get("title") or r.get("content")
                    ]
        except Exception as exc:
            logger.warning("SearXNG search failed for query %r: %s", query, exc)
            return []

    async def search_many(self, queries: list[str]) -> list[dict]:
        """Run multiple queries in parallel. Returns combined results list."""
        if not queries:
            return []
        results_per_query = await asyncio.gather(*[self.search(q) for q in queries])
        combined = []
        for results in results_per_query:
            combined.extend(results)
        return combined
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_search_service.py -v --timeout=5
```

Expected: All 5 tests PASS.

**Step 6: Commit**

```bash
git add services/search_service.py tests/test_search_service.py
git commit -m "feat: add SearchService wrapping SearXNG JSON API"
```

---

### Task 4: Add search classifier to `LLMService` with failing tests first

**Files:**
- Modify: `services/llm_service.py`
- Create: `tests/test_search_classifier.py`

**Step 1: Write the failing tests**

Create `tests/test_search_classifier.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_search_classifier.py -v --timeout=5
```

Expected: `AttributeError: 'LLMService' object has no attribute '_classify_search_need'`

**Step 3: Add `_classify_search_need` to `LLMService`**

In `services/llm_service.py`, add this method to the `LLMService` class (after `_inject_behavior_rules`, around line 338):

```python
def _classify_search_need(self, user_message: str) -> list[str]:
    """Ask the LLM if this message needs web search. Returns list of query strings (max 3), or []."""
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a search query classifier. Determine if the user's message requires "
                "current, real-world, or recent information to answer accurately.\n"
                "If yes: return a JSON array of 1-3 specific search queries (strings) that would help answer it.\n"
                "If no: return the JSON value null.\n"
                "Return ONLY valid JSON. No explanation. No markdown.\n"
                "Examples:\n"
                '  "who won the super bowl this year?" -> ["Super Bowl 2026 winner", "Super Bowl LX result"]\n'
                '  "what is 2+2?" -> null\n'
                '  "hey how are you" -> null\n'
                '  "latest news on AI regulation" -> ["AI regulation news 2026", "AI laws passed 2026"]\n'
            ),
        },
        {"role": "user", "content": user_message},
    ]
    try:
        raw = self._complete_messages_sync(prompt, temperature=0.0, max_tokens=150)
        raw = raw.strip()
        if raw.lower() == "null":
            return []
        import json
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, str)][:3]
        return []
    except Exception as exc:
        logger.warning("Search classifier failed: %s", exc)
        return []
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_search_classifier.py -v --timeout=5
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add services/llm_service.py tests/test_search_classifier.py
git commit -m "feat: add _classify_search_need to LLMService"
```

---

### Task 5: Wire search into `_build_messages` with failing tests first

**Files:**
- Modify: `services/llm_service.py`
- Create: `tests/test_search_injection.py`

**Step 1: Write the failing tests**

Create `tests/test_search_injection.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_search_injection.py -v --timeout=5
```

Expected: `TypeError: _build_messages() got an unexpected keyword argument 'search_results'`

**Step 3: Update `_build_messages` signature and body**

In `services/llm_service.py`, modify `_build_messages` (line 288):

Add `search_results: list[dict] | None = None` to the parameter list:

```python
def _build_messages(
    self,
    user_display_name: str,
    user_message: str,
    memory: Dict[str, str],
    recent_messages: List[Tuple[str, str, str]],
    conversation_summary: str = "",
    intent_category: str = "",
    conversation_goal: str = "",
    response_mode: str = "",
    tool_context: str = "",
    search_results: list[dict] | None = None,
) -> List[Dict[str, str]]:
```

Then after the existing `if tool_context:` block (around line 312), add:

```python
if search_results:
    lines = ["[SEARCH RESULTS]"]
    for r in search_results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        lines.append(f"- {title}: {snippet} ({url})")
    preamble_parts.append("\n".join(lines))
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_search_injection.py -v --timeout=5
```

Expected: All 3 tests PASS.

**Step 5: Run existing tests to check nothing broke**

```bash
pytest tests/ -v --timeout=5 -x
```

Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add services/llm_service.py tests/test_search_injection.py
git commit -m "feat: inject search results into _build_messages system context"
```

---

### Task 6: Wire classifier + search into `generate_reply` flow

**Files:**
- Modify: `services/llm_service.py`
- Modify: `bot.py` or wherever `LLMService` is instantiated (to pass `SearchService`)

**Step 1: Update `LLMService.__init__` to accept a `SearchService`**

In `services/llm_service.py`, update `__init__` (line 263):

```python
def __init__(self, performance_tracker=None, model_runtime_service=None, behavior_rule_service=None, search_service=None):
    # ... existing init code ...
    self.search_service = search_service
```

**Step 2: Update `generate_reply` to run classifier + search before calling `_generate_reply_sync`**

Replace the `generate_reply` method (line 528) with:

```python
async def generate_reply(
    self,
    user_display_name: str,
    user_message: str,
    memory: Dict[str, str],
    recent_messages: List[Tuple[str, str, str]],
    conversation_summary: str = "",
    intent_category: str = "",
    conversation_goal: str = "",
    response_mode: str = "",
    tool_context: str = "",
    behavior_rules: List[str] | None = None,
) -> str:
    started_at = time.perf_counter()
    try:
        search_results = []
        if self.search_service is not None and SEARXNG_ENABLED:
            try:
                queries = await asyncio.to_thread(self._classify_search_need, user_message)
                if queries:
                    search_results = await self.search_service.search_many(queries)
            except Exception as exc:
                logger.warning("Search pipeline failed, continuing without results: %s", exc)

        return await asyncio.to_thread(
            self._generate_reply_sync,
            user_display_name,
            user_message,
            memory,
            recent_messages,
            conversation_summary,
            intent_category,
            conversation_goal,
            response_mode,
            tool_context,
            behavior_rules,
            search_results,
        )
    finally:
        if self.performance_tracker is not None:
            self.performance_tracker.record_service_call(
                "llm.generate_reply",
                (time.perf_counter() - started_at) * 1000,
            )
```

**Step 3: Update `_generate_reply_sync` to accept and pass `search_results`**

Add `search_results: list[dict] | None = None` parameter to `_generate_reply_sync` (line 462), then pass it to `_build_messages`:

```python
messages = self._inject_behavior_rules(self._build_messages(
    user_display_name=user_display_name,
    user_message=user_message,
    memory=memory,
    recent_messages=recent_messages,
    conversation_summary=conversation_summary,
    intent_category=intent_category,
    conversation_goal=conversation_goal,
    response_mode=response_mode,
    tool_context=tool_context,
    search_results=search_results,
), behavior_rules)
```

**Step 4: Add `SEARXNG_ENABLED` import to `llm_service.py`**

In the imports section at the top of `services/llm_service.py`, add `SEARXNG_ENABLED` to the `core.config` import:

```python
from core.config import (
    ...
    SEARXNG_ENABLED,
    ...
)
```

**Step 5: Find where `LLMService` is instantiated and inject `SearchService`**

Search for instantiation:

```bash
grep -rn "LLMService(" --include="*.py" .
```

In whatever file constructs `LLMService`, add:

```python
from services.search_service import SearchService
from core.config import SEARXNG_BASE_URL, SEARXNG_MAX_RESULTS, SEARXNG_ENABLED

search_service = SearchService(base_url=SEARXNG_BASE_URL, max_results=SEARXNG_MAX_RESULTS) if SEARXNG_ENABLED else None
llm_service = LLMService(..., search_service=search_service)
```

**Step 6: Run all tests**

```bash
pytest tests/ -v --timeout=5 -x
```

Expected: All tests pass.

**Step 7: Commit**

```bash
git add services/llm_service.py
git commit -m "feat: wire web search classifier and SearXNG into generate_reply"
```

---

### Task 7: Smoke test end-to-end

**Step 1: Ensure SearXNG is running**

```bash
docker ps | grep searxng
```

Expected: Container listed and running.

**Step 2: Start the bot**

```bash
python bot.py
```

**Step 3: Ask a current-events question in Discord**

Send: `who won the most recent Super Bowl?`

Expected behavior:
- Bot pauses briefly (classifier call + search)
- Bot answers with information from search results, not hallucinated data

**Step 4: Ask a casual question**

Send: `hey what's your favorite color`

Expected behavior:
- Bot responds at normal speed (no search delay)

**Step 5: Stop SearXNG and ask a current-events question**

```bash
docker stop searxng
```

Send: `latest news on AI?`

Expected: Bot still responds (falls back gracefully, no crash, no error message to user).

**Step 6: Restart SearXNG**

```bash
docker start searxng
```

**Step 7: Commit nothing** — this is a verification step only.

---

### Task 8: Final cleanup commit

**Step 1: Run full test suite one more time**

```bash
pytest tests/ -v --timeout=5
```

Expected: All tests pass.

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: complete web search RAG via SearXNG — classifier + parallel search + context injection"
```
