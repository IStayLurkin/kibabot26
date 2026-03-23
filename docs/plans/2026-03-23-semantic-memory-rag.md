# Semantic Memory (Layer 1 RAG) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic/episodic memory using sqlite-vec + nomic-embed-text so the bot can recall past conversations and context by similarity — complementing the existing exact-match KV store.

**Architecture:** A new `VectorMemoryService` embeds and stores episodic memories (full sentences like "Brandon is building a Discord bot in Python") in a `vector_memories` SQLite table using sqlite-vec for cosine similarity search. On each chat turn, the user message is embedded and the top-5 most relevant past memories are retrieved and injected into the LLM context as a `[RELEVANT MEMORIES]` block. After each turn, a background task asks the LLM to extract any episodic content worth storing. The existing KV `user_memory` table is unchanged.

**Tech Stack:** Python 3.12, sqlite-vec (SQLite extension, pip install sqlite-vec), aiosqlite (already present), httpx for Ollama embedding API, nomic-embed-text via Ollama (already running at http://localhost:11434).

---

### Task 1: Install sqlite-vec and pull nomic-embed-text

**Files:**
- No code changes — setup only

**Step 1: Install sqlite-vec**

```bash
.venv/Scripts/pip.exe install sqlite-vec
```

**Step 2: Verify installation**

```bash
.venv/Scripts/python.exe -c "import sqlite_vec; print(sqlite_vec.loadable_path())"
```

Expected: prints a path ending in `.pyd` or `.so` — no error.

**Step 3: Pull nomic-embed-text via Ollama**

```bash
ollama pull nomic-embed-text
```

Expected: downloads model, ends with "success". If already present: "up to date".

**Step 4: Verify embedding works**

```bash
curl -s http://localhost:11434/api/embeddings -d "{\"model\":\"nomic-embed-text\",\"prompt\":\"hello world\"}" | python -c "import sys,json; d=json.load(sys.stdin); print(len(d['embedding']), 'dims')"
```

Expected: `768 dims` (nomic-embed-text produces 768-dimensional vectors).

**Step 5: No commit** — setup only.

---

### Task 2: Create `EmbeddingService` with TDD

**Files:**
- Create: `services/embedding_service.py`
- Create: `tests/test_embedding_service.py`

**Step 1: Write the failing tests**

Create `tests/test_embedding_service.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.embedding_service import EmbeddingService


@pytest.fixture
def svc():
    return EmbeddingService(base_url="http://localhost:11434", model="nomic-embed-text")


@pytest.mark.asyncio
async def test_embed_returns_list_of_floats(svc):
    fake_response = {"embedding": [0.1, 0.2, 0.3] * 256}  # 768 dims
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=fake_response)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        result = await svc.embed("hello world")

    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embed_returns_empty_on_error(svc):
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        result = await svc.embed("hello world")

    assert result == []


@pytest.mark.asyncio
async def test_embed_many_returns_list_of_embeddings(svc):
    fake_embedding = [0.1, 0.2, 0.3] * 256
    fake_response = {"embedding": fake_embedding}
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=fake_response)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        results = await svc.embed_many(["hello", "world"])

    assert len(results) == 2
    assert all(len(e) == 768 for e in results)
```

**Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_embedding_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'services.embedding_service'`

**Step 3: Install httpx if not present**

```bash
.venv/Scripts/pip.exe install httpx
```

**Step 4: Create `services/embedding_service.py`**

```python
import logging
import httpx

logger = logging.getLogger(__name__)

_EMBED_TIMEOUT = 10.0


class EmbeddingService:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns list of floats, or [] on failure."""
        try:
            async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
        except Exception as exc:
            logger.warning("[embedding] Failed to embed text: %s", exc)
            return []

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially. Returns list of embeddings (empty list on failure)."""
        results = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
        return results
```

**Step 5: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_embedding_service.py -v
```

Expected: 3 passed.

**Step 6: Commit**

```bash
git add services/embedding_service.py tests/test_embedding_service.py
git commit -m "feat: add EmbeddingService wrapping Ollama /api/embeddings"
```

---

### Task 3: Create vector_memories DB table

**Files:**
- Modify: `database/chat_memory.py`
- Create: `tests/test_vector_memory_db.py`

**Step 1: Write the failing test**

Create `tests/test_vector_memory_db.py`:

```python
import pytest
import aiosqlite
import sqlite_vec
import struct
from database.vector_memory_db import init_vector_memory_db, store_vector_memory, get_all_vector_memories


@pytest.fixture
async def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.enable_load_extension(True)
    await conn.load_extension(sqlite_vec.loadable_path())
    await init_vector_memory_db(conn)
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_store_and_retrieve_memory(db):
    embedding = [0.1] * 768
    await store_vector_memory(db, user_id="123", content="Brandon builds Discord bots", embedding=embedding)
    rows = await get_all_vector_memories(db, user_id="123")
    assert len(rows) == 1
    assert rows[0]["content"] == "Brandon builds Discord bots"


@pytest.mark.asyncio
async def test_store_multiple_memories(db):
    embedding = [0.1] * 768
    await store_vector_memory(db, user_id="123", content="Memory one", embedding=embedding)
    await store_vector_memory(db, user_id="123", content="Memory two", embedding=embedding)
    rows = await get_all_vector_memories(db, user_id="123")
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_memories_isolated_by_user(db):
    embedding = [0.1] * 768
    await store_vector_memory(db, user_id="AAA", content="User A memory", embedding=embedding)
    await store_vector_memory(db, user_id="BBB", content="User B memory", embedding=embedding)
    rows_a = await get_all_vector_memories(db, user_id="AAA")
    rows_b = await get_all_vector_memories(db, user_id="BBB")
    assert len(rows_a) == 1
    assert len(rows_b) == 1
    assert rows_a[0]["content"] == "User A memory"
```

**Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_vector_memory_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'database.vector_memory_db'`

**Step 3: Create `database/vector_memory_db.py`**

```python
import struct
import logging
import aiosqlite
import sqlite_vec

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


async def init_vector_memory_db(db: aiosqlite.Connection) -> None:
    """Create vector_memories table. Call after loading sqlite_vec extension."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS vector_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_vector_memories_user
        ON vector_memories(user_id)
    """)
    await db.commit()


async def store_vector_memory(
    db: aiosqlite.Connection,
    user_id: str,
    content: str,
    embedding: list[float],
) -> None:
    """Store a memory with its embedding blob."""
    blob = _pack_embedding(embedding)
    await db.execute(
        "INSERT INTO vector_memories (user_id, content, embedding) VALUES (?, ?, ?)",
        (user_id, content, blob),
    )
    await db.commit()


async def get_all_vector_memories(
    db: aiosqlite.Connection,
    user_id: str,
) -> list[aiosqlite.Row]:
    """Return all stored memories for a user (content + embedding blob)."""
    cursor = await db.execute(
        "SELECT id, content, embedding, created_at FROM vector_memories WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    )
    return await cursor.fetchall()


async def delete_vector_memories(db: aiosqlite.Connection, user_id: str) -> None:
    """Delete all vector memories for a user."""
    await db.execute("DELETE FROM vector_memories WHERE user_id = ?", (user_id,))
    await db.commit()
```

**Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_vector_memory_db.py -v
```

Expected: 3 passed.

**Step 5: Commit**

```bash
git add database/vector_memory_db.py tests/test_vector_memory_db.py
git commit -m "feat: add vector_memories table and DB helpers"
```

---

### Task 4: Create `VectorMemoryService` with cosine similarity retrieval and TDD

**Files:**
- Create: `services/vector_memory_service.py`
- Create: `tests/test_vector_memory_service.py`

**Step 1: Write the failing tests**

Create `tests/test_vector_memory_service.py`:

```python
import pytest
import math
from unittest.mock import AsyncMock, MagicMock, patch
from services.vector_memory_service import VectorMemoryService, _cosine_similarity


def test_cosine_similarity_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine_similarity(a, b) == 0.0


@pytest.mark.asyncio
async def test_store_memory_calls_embed_and_db():
    mock_embed_svc = AsyncMock()
    mock_embed_svc.embed = AsyncMock(return_value=[0.1] * 768)
    mock_db = AsyncMock()

    with patch("services.vector_memory_service.store_vector_memory", new_callable=AsyncMock) as mock_store:
        svc = VectorMemoryService(embedding_service=mock_embed_svc)
        await svc.store(mock_db, user_id="123", content="Brandon loves Python")

    mock_embed_svc.embed.assert_called_once_with("Brandon loves Python")
    mock_store.assert_called_once()


@pytest.mark.asyncio
async def test_store_memory_skips_on_empty_embedding():
    mock_embed_svc = AsyncMock()
    mock_embed_svc.embed = AsyncMock(return_value=[])  # embed failed
    mock_db = AsyncMock()

    with patch("services.vector_memory_service.store_vector_memory", new_callable=AsyncMock) as mock_store:
        svc = VectorMemoryService(embedding_service=mock_embed_svc)
        await svc.store(mock_db, user_id="123", content="some text")

    mock_store.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_returns_top_k_by_similarity():
    import struct

    def pack(v):
        return struct.pack(f"{len(v)}f", *v)

    # Query vector pointing in direction of memory_a
    query_vec = [1.0, 0.0] + [0.0] * 766
    memory_a_vec = [1.0, 0.0] + [0.0] * 766   # cos sim = 1.0 (identical)
    memory_b_vec = [0.0, 1.0] + [0.0] * 766   # cos sim = 0.0 (orthogonal)

    class FakeRow:
        def __init__(self, content, embedding_blob):
            self._data = {"content": content, "embedding": embedding_blob}
        def __getitem__(self, key):
            return self._data[key]

    fake_rows = [
        FakeRow("Memory A", pack(memory_a_vec)),
        FakeRow("Memory B", pack(memory_b_vec)),
    ]

    mock_embed_svc = AsyncMock()
    mock_embed_svc.embed = AsyncMock(return_value=query_vec)
    mock_db = AsyncMock()

    with patch("services.vector_memory_service.get_all_vector_memories", new_callable=AsyncMock, return_value=fake_rows):
        svc = VectorMemoryService(embedding_service=mock_embed_svc, top_k=1)
        results = await svc.retrieve(mock_db, user_id="123", query="some query")

    assert len(results) == 1
    assert results[0] == "Memory A"
```

**Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_vector_memory_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'services.vector_memory_service'`

**Step 3: Create `services/vector_memory_service.py`**

```python
import math
import struct
import logging

from database.vector_memory_db import store_vector_memory, get_all_vector_memories

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class VectorMemoryService:
    def __init__(self, embedding_service, top_k: int = 5):
        self._embed = embedding_service
        self._top_k = top_k

    async def store(self, db, user_id: str, content: str) -> None:
        """Embed content and store in vector_memories. Silently skips on embed failure."""
        embedding = await self._embed.embed(content)
        if not embedding:
            logger.warning("[vector_memory] Skipping store — embed returned empty for user %s", user_id)
            return
        await store_vector_memory(db, user_id=user_id, content=content, embedding=embedding)

    async def retrieve(self, db, user_id: str, query: str) -> list[str]:
        """Embed query and return top-K most similar memory contents. Returns [] on failure."""
        try:
            query_vec = await self._embed.embed(query)
            if not query_vec:
                return []
            rows = await get_all_vector_memories(db, user_id=user_id)
            if not rows:
                return []
            scored = []
            for row in rows:
                blob = row["embedding"]
                vec = _unpack_embedding(blob)
                score = _cosine_similarity(query_vec, vec)
                scored.append((score, row["content"]))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [content for _, content in scored[: self._top_k]]
        except Exception as exc:
            logger.warning("[vector_memory] Retrieval failed: %s", exc)
            return []
```

**Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_vector_memory_service.py -v
```

Expected: 8 passed.

**Step 5: Commit**

```bash
git add services/vector_memory_service.py tests/test_vector_memory_service.py
git commit -m "feat: add VectorMemoryService with cosine similarity retrieval"
```

---

### Task 5: Wire sqlite-vec extension into DB connection and init

**Files:**
- Modify: `database/db_connection.py`
- Modify: `database/chat_memory.py` (add `init_vector_memory_db` call)

**Step 1: Update `database/db_connection.py` to load sqlite-vec**

The current `get_db()` function (line 7) opens a plain aiosqlite connection. We need to load the sqlite-vec extension after opening.

Replace the file content:

```python
import aiosqlite
import sqlite_vec

DB_PATH = "bot.db"
_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    global _connection
    if _connection is None:
        _connection = await aiosqlite.connect(DB_PATH)
        await _connection.enable_load_extension(True)
        await _connection.load_extension(sqlite_vec.loadable_path())
        await _connection.enable_load_extension(False)
        await _connection.execute("PRAGMA journal_mode=WAL")
        await _connection.execute("PRAGMA foreign_keys = ON")
        await _connection.execute("PRAGMA synchronous = NORMAL")
        _connection.row_factory = aiosqlite.Row
    return _connection


async def close_db():
    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None
```

**Step 2: Add `init_vector_memory_db` call to `database/chat_memory.py`**

At the top of `chat_memory.py`, add import (line 2):
```python
from database.vector_memory_db import init_vector_memory_db
```

At the end of `init_chat_memory_db()`, before `await db.commit()` (line 80), add:
```python
    await init_vector_memory_db(db)
```

Remove the `await db.commit()` already in `init_vector_memory_db` to avoid double-commit — or just leave both; it's harmless.

**Step 3: Run existing tests to verify nothing broke**

```bash
.venv/Scripts/python.exe -m pytest tests/test_vector_memory_db.py tests/test_vector_memory_service.py tests/test_search_service.py tests/test_virustotal_service.py -v
```

Expected: all pass.

**Step 4: Commit**

```bash
git add database/db_connection.py database/chat_memory.py
git commit -m "feat: load sqlite-vec extension in DB connection, init vector_memories table on startup"
```

---

### Task 6: Add episodic memory extraction to `memory_service.py` with TDD

**Files:**
- Modify: `services/memory_service.py`
- Create: `tests/test_vector_memory_extraction.py`

This task adds `maybe_store_episodic_memory(llm, vector_memory_svc, db, user_id, user_message, bot_reply)` — called as a background task after each turn, similar to `maybe_update_summary`.

**Step 1: Write the failing tests**

Create `tests/test_vector_memory_extraction.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.memory_service import maybe_store_episodic_memory


@pytest.mark.asyncio
async def test_stores_memory_when_llm_says_yes():
    mock_llm = MagicMock()
    mock_llm.extract_episodic_memory = AsyncMock(return_value={
        "should_store": True,
        "content": "Brandon is building a Discord bot in Python"
    })
    mock_vms = AsyncMock()
    mock_db = AsyncMock()

    await maybe_store_episodic_memory(
        llm=mock_llm,
        vector_memory_service=mock_vms,
        db=mock_db,
        user_id="123",
        user_message="I'm building a Discord bot",
        bot_reply="That's cool!"
    )

    mock_vms.store.assert_called_once_with(mock_db, user_id="123", content="Brandon is building a Discord bot in Python")


@pytest.mark.asyncio
async def test_skips_when_llm_says_no():
    mock_llm = MagicMock()
    mock_llm.extract_episodic_memory = AsyncMock(return_value={"should_store": False, "content": ""})
    mock_vms = AsyncMock()
    mock_db = AsyncMock()

    await maybe_store_episodic_memory(
        llm=mock_llm,
        vector_memory_service=mock_vms,
        db=mock_db,
        user_id="123",
        user_message="hey",
        bot_reply="hey"
    )

    mock_vms.store.assert_not_called()


@pytest.mark.asyncio
async def test_skips_on_llm_exception():
    mock_llm = MagicMock()
    mock_llm.extract_episodic_memory = AsyncMock(side_effect=RuntimeError("LLM down"))
    mock_vms = AsyncMock()
    mock_db = AsyncMock()

    # Should not raise
    await maybe_store_episodic_memory(
        llm=mock_llm,
        vector_memory_service=mock_vms,
        db=mock_db,
        user_id="123",
        user_message="some text",
        bot_reply="some reply"
    )

    mock_vms.store.assert_not_called()


@pytest.mark.asyncio
async def test_skips_short_messages():
    mock_llm = MagicMock()
    mock_llm.extract_episodic_memory = AsyncMock()
    mock_vms = AsyncMock()
    mock_db = AsyncMock()

    await maybe_store_episodic_memory(
        llm=mock_llm,
        vector_memory_service=mock_vms,
        db=mock_db,
        user_id="123",
        user_message="ok",
        bot_reply="ok"
    )

    mock_llm.extract_episodic_memory.assert_not_called()
    mock_vms.store.assert_not_called()
```

**Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_vector_memory_extraction.py -v
```

Expected: `ImportError: cannot import name 'maybe_store_episodic_memory'`

**Step 3: Add `maybe_store_episodic_memory` to `services/memory_service.py`**

Append to the end of `services/memory_service.py`:

```python
async def maybe_store_episodic_memory(
    llm,
    vector_memory_service,
    db,
    user_id: str,
    user_message: str,
    bot_reply: str,
) -> None:
    """
    After a chat turn, ask the LLM if anything is worth storing as an episodic memory.
    Runs as a background task — never raises.
    """
    if len(user_message.split()) < 3:
        return
    try:
        result = await llm.extract_episodic_memory(
            user_message=user_message,
            bot_reply=bot_reply,
        )
        if not isinstance(result, dict) or not result.get("should_store"):
            return
        content = str(result.get("content", "")).strip()
        if not content:
            return
        await vector_memory_service.store(db, user_id=user_id, content=content)
    except Exception as exc:
        logger.warning("[episodic_memory] Extraction failed: %s", exc)
```

**Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_vector_memory_extraction.py -v
```

Expected: 4 passed.

**Step 5: Commit**

```bash
git add services/memory_service.py tests/test_vector_memory_extraction.py
git commit -m "feat: add maybe_store_episodic_memory to memory_service"
```

---

### Task 7: Add `extract_episodic_memory` to `LLMService` with TDD

**Files:**
- Modify: `services/llm_service.py`
- Create: `tests/test_episodic_memory_extraction.py`

**Step 1: Write the failing tests**

Create `tests/test_episodic_memory_extraction.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_episodic_memory_extraction.py -v
```

Expected: `AttributeError: 'LLMService' object has no attribute 'extract_episodic_memory_sync'`

**Step 3: Add `extract_episodic_memory_sync` and `extract_episodic_memory` to `LLMService`**

In `services/llm_service.py`, add after the `_classify_search_need` method (around line 406):

```python
def extract_episodic_memory_sync(self, user_message: str, bot_reply: str) -> dict:
    """
    Synchronous LLM call to decide if a conversation turn contains episodic content worth storing.
    Returns {"should_store": bool, "content": str}.
    """
    import json
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a memory curator. Given a conversation turn, decide if it contains "
                "a personal fact, preference, project, or ongoing context about the user that would "
                "be useful to remember in future conversations.\n"
                "If yes: return JSON {\"should_store\": true, \"content\": \"<one sentence summary of the fact>\"}\n"
                "If no: return JSON {\"should_store\": false, \"content\": \"\"}\n"
                "Return ONLY valid JSON. No explanation. No markdown.\n"
                "Examples:\n"
                "  User: 'I'm building a Discord bot in Python' -> {\"should_store\": true, \"content\": \"Brandon is building a Discord bot in Python\"}\n"
                "  User: 'hey what's up' -> {\"should_store\": false, \"content\": \"\"}\n"
                "  User: 'I prefer dark mode always' -> {\"should_store\": true, \"content\": \"Brandon prefers dark mode\"}\n"
            ),
        },
        {"role": "user", "content": f"User said: {user_message}\nBot replied: {bot_reply}"},
    ]
    try:
        raw = self._complete_messages_sync(prompt, temperature=0.0, max_tokens=100)
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict) and "should_store" in parsed:
            return parsed
        return {"should_store": False, "content": ""}
    except Exception as exc:
        logger.warning("[episodic_memory] LLM extraction failed: %s", exc)
        return {"should_store": False, "content": ""}

async def extract_episodic_memory(self, user_message: str, bot_reply: str) -> dict:
    """Async wrapper for extract_episodic_memory_sync."""
    return await asyncio.to_thread(self.extract_episodic_memory_sync, user_message, bot_reply)
```

**Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_episodic_memory_extraction.py -v
```

Expected: 4 passed.

**Step 5: Commit**

```bash
git add services/llm_service.py tests/test_episodic_memory_extraction.py
git commit -m "feat: add extract_episodic_memory to LLMService"
```

---

### Task 8: Wire `[RELEVANT MEMORIES]` into `_build_messages` with TDD

**Files:**
- Modify: `services/llm_service.py`
- Modify: `tests/test_search_injection.py` (extend existing)

**Step 1: Write the failing test**

Add to `tests/test_search_injection.py`:

```python
def test_build_messages_injects_relevant_memories(llm):
    memories = ["Brandon is building a Discord bot", "Brandon prefers dark mode"]
    messages = llm._build_messages(
        user_display_name="Brandon",
        user_message="what am I working on?",
        memory={},
        recent_messages=[],
        relevant_memories=memories,
    )
    system_content = messages[0]["content"]
    assert "RELEVANT MEMORIES" in system_content
    assert "Brandon is building a Discord bot" in system_content
    assert "Brandon prefers dark mode" in system_content


def test_build_messages_no_relevant_memories_no_block(llm):
    messages = llm._build_messages(
        user_display_name="Brandon",
        user_message="hey",
        memory={},
        recent_messages=[],
        relevant_memories=[],
    )
    system_content = messages[0]["content"]
    assert "RELEVANT MEMORIES" not in system_content
```

**Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_search_injection.py -v
```

Expected: `TypeError: _build_messages() got an unexpected keyword argument 'relevant_memories'`

**Step 3: Update `_build_messages` in `services/llm_service.py`**

Add `relevant_memories: list[str] | None = None` to the parameter list (after `search_results`):

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
    relevant_memories: list[str] | None = None,
) -> List[Dict[str, str]]:
```

After the `if search_results:` block (around line 355), add:

```python
if relevant_memories:
    lines = ["[RELEVANT MEMORIES]"]
    for m in relevant_memories:
        lines.append(f"- {m}")
    preamble_parts.append("\n".join(lines))
```

**Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_search_injection.py -v
```

Expected: all pass (5 total now).

**Step 5: Commit**

```bash
git add services/llm_service.py tests/test_search_injection.py
git commit -m "feat: inject [RELEVANT MEMORIES] block into _build_messages"
```

---

### Task 9: Wire `relevant_memories` through `_generate_reply_sync` and `generate_reply`

**Files:**
- Modify: `services/llm_service.py`

This mirrors what was done for `search_results` in the web search feature. No new tests needed — existing injection tests already cover the block content.

**Step 1: Add `relevant_memories` to `_generate_reply_sync`**

In `_generate_reply_sync` (around line 535), add parameter and pass-through:

```python
def _generate_reply_sync(
    self,
    ...
    search_results: list[dict] | None = None,
    relevant_memories: list[str] | None = None,
) -> str:
    messages = self._inject_behavior_rules(self._build_messages(
        ...
        search_results=search_results,
        relevant_memories=relevant_memories,
    ), behavior_rules)
```

**Step 2: Add retrieval + injection in `generate_reply`**

`generate_reply` receives `vector_memory_service` and `db` via the service — but these aren't on `LLMService` currently. The cleanest approach: pass `relevant_memories` in from the call site (`chat_service.py`), not from inside `generate_reply`. So `generate_reply` just needs to accept and thread it through.

Add `relevant_memories: list[str] | None = None` to `generate_reply` signature and pass to `_generate_reply_sync`:

```python
async def generate_reply(
    self,
    ...
    relevant_memories: list[str] | None = None,
) -> str:
    ...
    return await asyncio.to_thread(
        self._generate_reply_sync,
        ...
        search_results,
        relevant_memories,
    )
```

**Step 3: Run full relevant test suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_search_injection.py tests/test_search_classifier.py tests/test_episodic_memory_extraction.py tests/test_vector_memory_service.py -v
```

Expected: all pass.

**Step 4: Commit**

```bash
git add services/llm_service.py
git commit -m "feat: thread relevant_memories through _generate_reply_sync and generate_reply"
```

---

### Task 10: Wire everything into `chat_service.py` and `bot.py`

**Files:**
- Modify: `services/chat_service.py`
- Modify: `bot.py`

This is the final wiring step — no new tests (the unit tests already cover each piece).

**Step 1: Update `bot.py` to construct `EmbeddingService` and `VectorMemoryService`**

In `bot.py`, add imports:

```python
from services.embedding_service import EmbeddingService
from services.vector_memory_service import VectorMemoryService
```

In `setup_hook`, after `self.llm_service = LLMService(...)`:

```python
embedding_service = EmbeddingService(base_url=OLLAMA_BASE_URL.replace("/v1", ""), model="nomic-embed-text")
self.vector_memory_service = VectorMemoryService(embedding_service=embedding_service, top_k=5)
```

Add `self.vector_memory_service = None` to `__init__` alongside other service declarations.

Note: `OLLAMA_BASE_URL` is `http://localhost:11434/v1` — strip the `/v1` for the embeddings endpoint.

**Step 2: Update `generate_dynamic_reply` in `chat_service.py`**

`generate_dynamic_reply` currently has this signature (around line 115):

```python
async def generate_dynamic_reply(
    user_id: str,
    channel_id: str,
    user_display_name: str,
    user_text: str,
    services: dict | None = None,
) -> ChatReply:
```

Add retrieval before the LLM call. Find the section where `memory`, `recent_messages`, and `conversation_summary` are retrieved (around line 135-145). After those, add:

```python
# Semantic memory retrieval
relevant_memories = []
vector_memory_service = (services or {}).get("vector_memory_service")
db_conn = None
try:
    if vector_memory_service is not None:
        from database.db_connection import get_db
        db_conn = await get_db()
        relevant_memories = await vector_memory_service.retrieve(db_conn, user_id=user_id, query=user_text)
except Exception as exc:
    logger.warning("[vector_memory] Retrieval failed in chat_service: %s", exc)
```

Then pass `relevant_memories` to `llm.generate_reply(...)` — find the call (around line 235+) and add:

```python
reply_text = await llm.generate_reply(
    ...
    relevant_memories=relevant_memories,
)
```

**Step 3: Add background episodic memory storage after the reply**

After `add_chat_message(session_id, "bot", reply.content)`, add:

```python
# Background: store episodic memory
if vector_memory_service is not None and db_conn is not None:
    asyncio.create_task(
        maybe_store_episodic_memory(
            llm=llm,
            vector_memory_service=vector_memory_service,
            db=db_conn,
            user_id=user_id,
            user_message=user_text,
            bot_reply=reply.content,
        )
    )
```

Add import at top of `chat_service.py`:

```python
from services.memory_service import format_memory, maybe_store_episodic_memory
```

**Step 4: Pass `vector_memory_service` from `cogs/chat_commands.py`**

Find where `services` dict is constructed in `chat_commands.py` (the dict passed to `generate_dynamic_reply`). Add:

```python
"vector_memory_service": getattr(bot, "vector_memory_service", None),
```

**Step 5: Run full test suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_search_injection.py tests/test_search_classifier.py tests/test_episodic_memory_extraction.py tests/test_vector_memory_service.py tests/test_vector_memory_db.py tests/test_embedding_service.py -v
```

Expected: all pass.

**Step 6: Commit**

```bash
git add services/chat_service.py bot.py cogs/chat_commands.py
git commit -m "feat: wire semantic memory retrieval and episodic storage into chat pipeline"
```

---

### Task 11: Smoke test end-to-end

**Step 1: Start the bot**

```bash
.venv/Scripts/python.exe bot.py
```

Watch logs for:
- `[ollama] Already running` or `[ollama] Ready`
- No errors loading extensions
- Bot connects to Discord

**Step 2: Send a personal fact**

In Discord: `I'm building a Discord bot called Kiba in Python`

Expected: bot replies normally. No visible change yet (memory stored in background).

**Step 3: Ask a follow-up in a fresh conversation**

In Discord: `what projects am I working on?`

Expected: bot references the Discord bot — retrieved from vector memory, not just recent history.

**Step 4: Verify memory was stored**

```bash
.venv/Scripts/python.exe -c "
import asyncio, aiosqlite, sqlite_vec

async def check():
    db = await aiosqlite.connect('bot.db')
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    cur = await db.execute('SELECT user_id, content, created_at FROM vector_memories ORDER BY id DESC LIMIT 10')
    rows = await cur.fetchall()
    for r in rows:
        print(r)
    await db.close()

asyncio.run(check())
"
```

Expected: rows with your user_id and the extracted memory content.

**Step 5: Commit nothing** — verification only.

---

## Notes

- **nomic-embed-text** produces 768-dim vectors. If you switch models, update `EMBEDDING_DIM = 768` in `database/vector_memory_db.py`.
- **Cosine similarity is computed in Python** (not SQL) for simplicity — with <1000 memories per user this is fast enough. If you accumulate thousands, consider adding an ANN index.
- **EmbeddingService** is synchronous-friendly via `asyncio.to_thread` if needed, but uses httpx async natively.
- **Memory deduplication** is not implemented — the LLM extractor will sometimes store similar facts. This can be added later by checking cosine similarity against existing memories before storing.
- **OLLAMA_BASE_URL** in config is `http://localhost:11434/v1` (OpenAI-compat). The embeddings endpoint is `http://localhost:11434/api/embeddings` (native Ollama). Strip `/v1` when constructing `EmbeddingService`.
