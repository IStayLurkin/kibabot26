# Kiba Bot: Performance & Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 15 improvements covering DB connection pooling, threading fixes, async fixes, streaming replies, circuit breaker, prompt enhancement, and new user-facing commands.

**Architecture:** Single shared aiosqlite connection with WAL mode replaces per-call connections. A dedicated `ThreadPoolExecutor` for heavy jobs (image/music) prevents thread starvation on chat. Streaming LLM replies use Discord message edits. A circuit breaker on Ollama tracks failure count and skips it for a cooldown period. New commands (`!forget`, `!models`, `!purge`) are added to existing cogs.

**Tech Stack:** Python 3.12, discord.py 2.7, aiosqlite, asyncio, concurrent.futures.ThreadPoolExecutor, Ollama streaming API (openai SDK stream=True)

---

## Task 1: SQLite WAL mode + shared connection

**Files:**
- Modify: `database/db_connection.py` (CREATE new file)
- Modify: `database/chat_memory.py`
- Modify: `database/database.py`
- Test: `tests/test_db_connection.py` (CREATE new file)

**Step 1: Write the failing test**

```python
# tests/test_db_connection.py
import pytest
import asyncio
from database.db_connection import get_db, close_db, _connection

@pytest.mark.asyncio
async def test_get_db_returns_connection():
    db = await get_db()
    assert db is not None

@pytest.mark.asyncio
async def test_get_db_returns_same_instance():
    db1 = await get_db()
    db2 = await get_db()
    assert db1 is db2

@pytest.mark.asyncio
async def test_close_db_clears_connection():
    await get_db()
    await close_db()
    import database.db_connection as mod
    assert mod._connection is None
```

**Step 2: Run test to verify it fails**

```
cd G:\code\python\learn_python\bot\discord_bot_things
.venv\Scripts\python.exe -m pytest tests/test_db_connection.py -v
```
Expected: ImportError or FAIL — `db_connection` doesn't exist yet.

**Step 3: Create `database/db_connection.py`**

```python
import aiosqlite

DB_PATH = "bot.db"
_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    global _connection
    if _connection is None:
        _connection = await aiosqlite.connect(DB_PATH)
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

**Step 4: Run test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/test_db_connection.py -v
```
Expected: 3 PASS

**Step 5: Migrate `database/chat_memory.py` to use shared connection**

Replace every `async with aiosqlite.connect(DB_PATH) as db:` block. Each function should instead do:

```python
from database.db_connection import get_db

async def get_or_create_session(user_id: str, channel_id: str) -> int:
    db = await get_db()
    cursor = await db.execute(
        "SELECT id FROM chat_sessions WHERE user_id = ? AND channel_id = ?",
        (user_id, channel_id),
    )
    row = await cursor.fetchone()
    if row:
        return row[0]
    cursor = await db.execute(
        "INSERT INTO chat_sessions (user_id, channel_id) VALUES (?, ?)",
        (user_id, channel_id),
    )
    await db.commit()
    return cursor.lastrowid
```

Apply the same pattern to all other functions in `chat_memory.py` — remove the `async with aiosqlite.connect` wrapper from each, replace with `db = await get_db()`, keep the queries identical, keep `await db.commit()` after writes.

Also remove `await db.execute("PRAGMA foreign_keys = ON")` from individual functions — it's set once in `get_db()`.

**Step 6: Update `database/database.py` `init_db()`**

Find the `init_db()` function. After all table creation, add WAL mode setup:

```python
from database.db_connection import get_db

async def init_db():
    db = await get_db()           # initializes WAL + foreign_keys
    # ... existing CREATE TABLE statements but using `db` directly instead of a new connection
    await db.commit()
```

Check if `database.py` also opens its own connection — if so, migrate it to `get_db()` the same way.

**Step 7: Update `bot.py` close() to call close_db()**

In `ExpenseBot.close()` (`bot.py:194`):

```python
async def close(self):
    logger.info("Shutting down background tasks...")
    self.task_manager.stop_all()
    from database.db_connection import close_db
    await close_db()
    await super().close()
```

**Step 8: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 9: Commit**

```bash
git add database/db_connection.py database/chat_memory.py database/database.py bot.py tests/test_db_connection.py
git commit -m "perf: shared aiosqlite connection with WAL mode replaces per-call connections"
```

---

## Task 2: Fix `time.sleep()` → `await asyncio.sleep()` in ComfyUI poller

**Files:**
- Modify: `services/llm_service.py:1022-1036`

**Step 1: Open `services/llm_service.py` and locate `_poll_comfyui_history` at line 1022**

The method is synchronous (`def _poll_comfyui_history`). It is called from `_generate_comfyui_image_sync` which runs inside `asyncio.to_thread()`. The sleep is fine as a `time.sleep` here since we ARE in a thread — but the loop count (240 × 1s = 4 min max) is undocumented. Add a comment and a configurable timeout constant.

```python
# At top of file with other constants (around line 30-50):
COMFYUI_POLL_INTERVAL_SECONDS = 1
COMFYUI_POLL_MAX_ATTEMPTS = 240  # 4 minutes

# Replace the method:
def _poll_comfyui_history(self, prompt_id: str) -> dict:
    history_url = f"{COMFYUI_BASE_URL.rstrip('/')}/history/{prompt_id}"
    last_error = None
    for _attempt in range(COMFYUI_POLL_MAX_ATTEMPTS):
        try:
            history = self._get_json(history_url, timeout=30)
            if history and prompt_id in history:
                return history
        except Exception as exc:
            last_error = exc
        time.sleep(COMFYUI_POLL_INTERVAL_SECONDS)

    if last_error is not None:
        raise RuntimeError(f"ComfyUI history polling failed: {last_error}") from last_error
    raise RuntimeError("Timed out waiting for ComfyUI image generation.")
```

This is purely a documentation/maintainability fix — behavior is unchanged since this runs in a thread.

**Step 2: Commit**

```bash
git add services/llm_service.py
git commit -m "refactor: document ComfyUI poll timeout with named constants"
```

---

## Task 3: Dedicated ThreadPoolExecutor for heavy jobs

**Files:**
- Create: `core/executors.py`
- Modify: `services/image_service.py`
- Modify: `services/music_service.py`
- Test: `tests/test_executors.py`

**Step 1: Write failing test**

```python
# tests/test_executors.py
import pytest
from core.executors import HEAVY_EXECUTOR, LIGHT_EXECUTOR
from concurrent.futures import ThreadPoolExecutor

def test_heavy_executor_is_thread_pool():
    assert isinstance(HEAVY_EXECUTOR, ThreadPoolExecutor)

def test_light_executor_is_thread_pool():
    assert isinstance(LIGHT_EXECUTOR, ThreadPoolExecutor)

def test_executors_are_different_objects():
    assert HEAVY_EXECUTOR is not LIGHT_EXECUTOR
```

**Step 2: Run test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/test_executors.py -v
```
Expected: ImportError.

**Step 3: Create `core/executors.py`**

```python
from concurrent.futures import ThreadPoolExecutor

# For long-running jobs: image generation, music synthesis (15min YuE)
# Capped at 2 so we never OOM from parallel VRAM allocations
HEAVY_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kiba-heavy")

# For short blocking ops: nvidia-smi, DB init, short HTTP calls
LIGHT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kiba-light")
```

**Step 4: Run test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/test_executors.py -v
```
Expected: 3 PASS.

**Step 5: Use HEAVY_EXECUTOR in `services/image_service.py`**

Find `generate_image` and `generate_sdxl` methods. Each currently uses `asyncio.to_thread(...)`. Replace with `asyncio.get_event_loop().run_in_executor(HEAVY_EXECUTOR, ...)`:

```python
from core.executors import HEAVY_EXECUTOR

# In generate_image():
path = await asyncio.get_event_loop().run_in_executor(
    HEAVY_EXECUTOR, self._generate_sync_locked, prompt, progress_callback
)

# In generate_sdxl():
path = await asyncio.get_event_loop().run_in_executor(
    HEAVY_EXECUTOR, self._generate_sdxl_sync, prompt, progress_callback
)
```

**Step 6: Use HEAVY_EXECUTOR in `services/music_service.py`**

Find where `subprocess.run(...)` for YuE inference is called (around line 158). It should already be wrapped in `asyncio.to_thread`. Replace with `run_in_executor(HEAVY_EXECUTOR, ...)` for any calls that wrap `_generate_yue_studio` or similar long-running sync methods.

**Step 7: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 8: Commit**

```bash
git add core/executors.py services/image_service.py services/music_service.py tests/test_executors.py
git commit -m "perf: dedicated thread pool executors separate heavy jobs from chat/light ops"
```

---

## Task 4: Fix threading.Lock → asyncio.Lock in ImageService

**Files:**
- Modify: `services/image_service.py:35`

**Step 1: Write failing test**

```python
# Add to tests/test_executors.py or a new test_image_service.py
import pytest
import asyncio
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_image_service_lock_is_async():
    from services.image_service import ImageService
    svc = ImageService()
    # asyncio.Lock has acquire() coroutine; threading.Lock does not
    assert asyncio.iscoroutinefunction(svc._generation_lock.acquire) or hasattr(svc._generation_lock, '_waiters')
```

**Step 2: Run test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/ -k "test_image_service_lock_is_async" -v
```
Expected: FAIL — `threading.Lock` doesn't have `_waiters`.

**Step 3: Change the lock in `services/image_service.py`**

Line 35: change `self._generation_lock = threading.Lock()` to `self._generation_lock = asyncio.Lock()`

Then find all places `_generation_lock` is acquired. It's likely used in a `with self._generation_lock:` block inside a sync method. Since the sync method runs in a thread executor, we cannot use `async with` there.

The correct fix is to acquire the lock at the **async** level before dispatching to the executor:

```python
# In generate_image():
async def generate_image(self, prompt: str, progress_callback=None) -> str | None:
    async with self._generation_lock:
        return await asyncio.get_event_loop().run_in_executor(
            HEAVY_EXECUTOR, self._generate_flux_sync, prompt, progress_callback
        )

# In generate_sdxl():
async def generate_sdxl(self, prompt: str, progress_callback=None) -> str | None:
    async with self._generation_lock:
        return await asyncio.get_event_loop().run_in_executor(
            HEAVY_EXECUTOR, self._generate_sdxl_sync, prompt, progress_callback
        )
```

Remove `threading` import if it's no longer used. Remove any `with self._generation_lock:` inside the sync `_generate_*_sync` methods.

**Step 4: Run tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 5: Commit**

```bash
git add services/image_service.py
git commit -m "fix: replace threading.Lock with asyncio.Lock in ImageService to prevent async race conditions"
```

---

## Task 5: Ollama circuit breaker

**Files:**
- Create: `services/circuit_breaker.py`
- Modify: `services/llm_service.py` (around `_generate_reply_sync` and `_build_provider_chain`)
- Test: `tests/test_circuit_breaker.py`

**Step 1: Write failing test**

```python
# tests/test_circuit_breaker.py
import pytest
import time
from services.circuit_breaker import CircuitBreaker

def test_circuit_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)
    assert cb.is_available() is True

def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_available() is False

def test_circuit_resets_on_success():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.is_available() is True

def test_circuit_recovers_after_cooldown():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_available() is False
    time.sleep(0.01)
    assert cb.is_available() is True
```

**Step 2: Run test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/test_circuit_breaker.py -v
```
Expected: ImportError.

**Step 3: Create `services/circuit_breaker.py`**

```python
import time


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 120.0):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._opened_at: float | None = None

    def is_available(self) -> bool:
        if self._opened_at is None:
            return True
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self._cooldown_seconds:
            # Auto-reset: allow one probe attempt
            self._failure_count = 0
            self._opened_at = None
            return True
        return False

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._opened_at = time.monotonic()

    def record_success(self):
        self._failure_count = 0
        self._opened_at = None
```

**Step 4: Run test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/test_circuit_breaker.py -v
```
Expected: 4 PASS.

**Step 5: Wire circuit breaker into `LLMService`**

In `services/llm_service.py`, in `__init__`:

```python
from services.circuit_breaker import CircuitBreaker
# Add to __init__:
self._circuit_breakers: dict[str, CircuitBreaker] = {
    "ollama": CircuitBreaker(failure_threshold=3, cooldown_seconds=120),
    "hf": CircuitBreaker(failure_threshold=3, cooldown_seconds=120),
}
```

In `_build_provider_chain()`, filter out tripped providers:

```python
def _build_provider_chain(self) -> list[str]:
    # ... existing chain building logic ...
    chain = [p for p in chain if self._circuit_breakers.get(p, CircuitBreaker()).is_available()]
    return chain or ["ollama"]  # always have a fallback
```

In `_generate_reply_sync` (find the try/except that calls `_create_chat_completion`), wrap the success/failure recording:

```python
try:
    response = self._create_chat_completion(provider, ...)
    self._circuit_breakers[provider].record_success()
    return ...
except Exception as exc:
    self._circuit_breakers[provider].record_failure()
    logger.warning("[llm] provider=%s failed: %s", provider, exc)
    continue
```

**Step 6: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 7: Commit**

```bash
git add services/circuit_breaker.py services/llm_service.py tests/test_circuit_breaker.py
git commit -m "feat: Ollama/HF circuit breaker — skips failing provider for 2min after 3 failures"
```

---

## Task 6: Streaming LLM replies

**Files:**
- Modify: `services/llm_service.py` — add `generate_text_stream()` method
- Modify: `cogs/chat_commands.py` — use streaming in `handle_chat_turn`
- Test: `tests/test_streaming.py`

**Background:** Ollama's OpenAI-compatible API supports `stream=True`. Instead of waiting for the full response, we get token chunks via an async generator and edit the Discord message progressively. This makes the bot feel dramatically faster even if total generation time is the same.

**Step 1: Write failing test**

```python
# tests/test_streaming.py
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.mark.asyncio
async def test_generate_text_stream_yields_chunks():
    from services.llm_service import LLMService
    svc = LLMService()

    # Mock the OpenAI client stream
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " world"
    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta.content = None  # end of stream

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([chunk1, chunk2, chunk3]))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_stream

    with patch.object(svc, '_get_client_for_provider', return_value=mock_client):
        chunks = []
        async for chunk in svc.generate_text_stream("Hi"):
            chunks.append(chunk)

    assert chunks == ["Hello", " world"]
```

**Step 2: Run test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/test_streaming.py -v
```
Expected: AttributeError — `generate_text_stream` doesn't exist.

**Step 3: Add `generate_text_stream` to `services/llm_service.py`**

Add this method to `LLMService`. Place it near `generate_text`:

```python
async def generate_text_stream(self, prompt: str, *, history: list[dict] | None = None):
    """
    Async generator that yields text chunks as they arrive from Ollama.
    Falls back to a single yield of generate_text() if streaming fails.
    """
    messages = self._build_messages(prompt, history=history or [])
    provider_chain = self._build_provider_chain()
    provider = provider_chain[0]  # stream only from primary provider
    model = self._get_model_for_provider(provider)

    def _iter_stream():
        client = self._get_client_for_provider(provider)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    try:
        # Run the blocking stream iterator in a thread, yield chunks to async caller
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _produce():
            try:
                for text in _iter_stream():
                    loop.call_soon_threadsafe(queue.put_nowait, text)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        import concurrent.futures
        loop.run_in_executor(None, _produce)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
            self._circuit_breakers[provider].record_success()
    except Exception as exc:
        logger.warning("[llm] streaming failed, falling back to full generate: %s", exc)
        self._circuit_breakers[provider].record_failure()
        result = await self.generate_text(prompt, history=history)
        yield result
```

**Step 4: Run test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/test_streaming.py -v
```
Expected: PASS.

**Step 5: Use streaming in `cogs/chat_commands.py` `handle_chat_turn`**

Find the text path in `handle_chat_turn` (around line 262). The current flow is:
```python
reply = await generate_dynamic_reply(...)
if reply.content:
    await send_long_message(destination, reply.content)
```

Streaming works at the `generate_dynamic_reply` level, but that function does intent routing, memory lookup, etc. For now, stream only when intent is plain chat (not tools/draw/sing). The streaming happens inside `chat_service.generate_dynamic_reply`.

In `services/chat_service.py`, find where `llm.generate_text(...)` is called for plain chat replies. There will be a call like:
```python
raw = await llm.generate_text(user_text, history=history_lines, ...)
```

Replace it to accept an optional `message_ref` for editing:

Actually, the simpler approach: stream at the cog level for the final bot reply. In `handle_chat_turn`, after getting back `reply.content`, if the content is long (>200 chars), send a placeholder and stream edit:

```python
# In handle_chat_turn, text path:
reply = await generate_dynamic_reply(...)

if reply.content:
    await add_chat_message(session_id, "bot", reply.content)
    if len(reply.content) > 200:
        # Send placeholder, then "stream" by editing in chunks
        # (Ollama already returned full text — simulate streaming via chunked edits)
        placeholder = await destination.send("...")
        chunk_size = 200
        for i in range(0, len(reply.content), chunk_size):
            await placeholder.edit(content=reply.content[:i + chunk_size])
            await asyncio.sleep(0.05)
    else:
        await send_long_message(destination, reply.content)
```

Note: True streaming requires refactoring `generate_dynamic_reply` to be an async generator — that's a larger change. The chunked-edit approach above gives the visual feel of streaming without the refactor. This is the YAGNI-compliant approach for now.

**Step 6: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 7: Commit**

```bash
git add services/llm_service.py cogs/chat_commands.py tests/test_streaming.py
git commit -m "feat: streaming-style reply delivery via chunked Discord message edits"
```

---

## Task 7: Image prompt enhancement via Ollama

**Files:**
- Modify: `services/image_service.py` — add `enhance_prompt()` call
- Modify: `cogs/chat_commands.py` — pass LLM service to image handler
- Test: `tests/test_prompt_enhancement.py`

**Step 1: Write failing test**

```python
# tests/test_prompt_enhancement.py
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_enhance_prompt_returns_string():
    from services.llm_service import LLMService
    svc = LLMService()
    svc.generate_text = AsyncMock(return_value="a majestic wolf in neon cyberpunk city, detailed, 8k")

    result = await svc.enhance_image_prompt("wolf city")
    assert isinstance(result, str)
    assert len(result) > len("wolf city")

@pytest.mark.asyncio
async def test_enhance_prompt_falls_back_on_error():
    from services.llm_service import LLMService
    svc = LLMService()
    svc.generate_text = AsyncMock(side_effect=Exception("LLM down"))

    result = await svc.enhance_image_prompt("wolf city")
    assert result == "wolf city"  # returns original on failure
```

**Step 2: Run test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/test_prompt_enhancement.py -v
```
Expected: AttributeError — `enhance_image_prompt` doesn't exist.

**Step 3: Add `enhance_image_prompt` to `services/llm_service.py`**

```python
async def enhance_image_prompt(self, prompt: str) -> str:
    """
    Asks Ollama to enrich a short image prompt with artistic detail.
    Returns the original prompt on any failure (never blocks generation).
    """
    instruction = (
        f"Rewrite this image generation prompt to be more detailed and vivid for a diffusion model. "
        f"Return ONLY the improved prompt, no explanation, no quotes.\n\nOriginal: {prompt}"
    )
    try:
        enhanced = await self.generate_text(instruction)
        enhanced = enhanced.strip().strip('"').strip("'")
        return enhanced if enhanced else prompt
    except Exception:
        return prompt
```

**Step 4: Run test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/test_prompt_enhancement.py -v
```
Expected: 2 PASS.

**Step 5: Wire into `handle_image_request` in `cogs/chat_commands.py`**

In `handle_image_request`, before calling `self.image_service.generate_image(...)`:

```python
async def handle_image_request(self, ctx, prompt: str, mode: str = "FLUX"):
    icon = "🎨" if mode == "FLUX" else "⚡"
    status_msg = await ctx.send(f"{icon} **Kiba is initializing {mode}...**\n[░░░░░░░░░░] 0%")

    # Enhance the prompt via Ollama (fast, ~200ms)
    enhanced_prompt = await self.llm.enhance_image_prompt(prompt)
    if enhanced_prompt != prompt:
        await status_msg.edit(content=f"{icon} **Prompt enhanced. Rendering ({mode})...**\n[░░░░░░░░░░] 0%")

    # Rest of the method uses enhanced_prompt instead of prompt
    ...
```

**Step 6: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 7: Commit**

```bash
git add services/llm_service.py cogs/chat_commands.py tests/test_prompt_enhancement.py
git commit -m "feat: auto-enhance image prompts via Ollama before FLUX/SDXL generation"
```

---

## Task 8: `!forget` command — clear user chat history

**Files:**
- Modify: `database/chat_memory.py` — add `delete_user_history()`
- Modify: `cogs/chat_commands.py` — add `!forget` command
- Test: `tests/test_forget_command.py`

**Step 1: Write failing test**

```python
# tests/test_forget_command.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_delete_user_history_runs_without_error():
    from database.chat_memory import delete_user_history
    # Should not raise even if user has no history
    await delete_user_history("999999", "888888")
```

**Step 2: Run test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/test_forget_command.py -v
```
Expected: ImportError — function doesn't exist.

**Step 3: Add `delete_user_history` to `database/chat_memory.py`**

```python
async def delete_user_history(user_id: str, channel_id: str):
    """Deletes all chat messages, session, summary, state, and memory for a user/channel."""
    db = await get_db()
    # Delete session (cascades to chat_messages via FK)
    await db.execute(
        "DELETE FROM chat_sessions WHERE user_id = ? AND channel_id = ?",
        (user_id, channel_id),
    )
    await db.execute(
        "DELETE FROM chat_summaries WHERE user_id = ? AND channel_id = ?",
        (user_id, channel_id),
    )
    await db.execute(
        "DELETE FROM chat_state WHERE user_id = ? AND channel_id = ?",
        (user_id, channel_id),
    )
    await db.execute(
        "DELETE FROM user_memory WHERE user_id = ?",
        (user_id,),
    )
    await db.commit()
```

**Step 4: Run test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/test_forget_command.py -v
```
Expected: PASS.

**Step 5: Add `!forget` command to `cogs/chat_commands.py`**

Add this command to the `ChatCommands` cog:

```python
@commands.command(name="forget")
async def forget_history(self, ctx):
    """Clears your entire chat history and memory with Kiba in this channel."""
    from database.chat_memory import delete_user_history
    user_id = str(ctx.author.id)
    channel_id = str(ctx.channel.id)
    await delete_user_history(user_id, channel_id)
    embed = discord.Embed(
        title="🧹 Memory Cleared",
        description="Your chat history and memory in this channel have been wiped. Starting fresh.",
        color=discord.Color.red(),
    )
    await ctx.send(embed=embed)
```

**Step 6: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 7: Commit**

```bash
git add database/chat_memory.py cogs/chat_commands.py tests/test_forget_command.py
git commit -m "feat: !forget command — users can wipe their own chat history and memory"
```

---

## Task 9: `!models` command — list loaded Ollama models

**Files:**
- Modify: `services/hardware_service.py` — add `get_ollama_running_models()`
- Modify: `cogs/chat_commands.py` — add `!models` command
- Test: `tests/test_models_command.py`

**Step 1: Write failing test**

```python
# tests/test_models_command.py
import pytest
from unittest.mock import patch, MagicMock

def test_get_ollama_running_models_returns_list():
    from services.hardware_service import HardwareService
    svc = HardwareService()
    with patch('urllib.request.urlopen') as mock_open:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"models": [{"name": "kiba:latest", "size": 9000000000}]}'
        mock_open.return_value = mock_resp
        result = svc.get_ollama_running_models()
    assert isinstance(result, list)
    assert result[0]["name"] == "kiba:latest"

def test_get_ollama_running_models_returns_empty_on_error():
    from services.hardware_service import HardwareService
    svc = HardwareService()
    with patch('urllib.request.urlopen', side_effect=Exception("connection refused")):
        result = svc.get_ollama_running_models()
    assert result == []
```

**Step 2: Run test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/test_models_command.py -v
```
Expected: AttributeError.

**Step 3: Add `get_ollama_running_models` to `services/hardware_service.py`**

```python
def get_ollama_running_models(self) -> list[dict]:
    """
    Returns models currently loaded in Ollama VRAM via /api/ps.
    Each entry has 'name' and 'size' (bytes).
    Returns [] on any failure.
    """
    from urllib.parse import urlparse
    parsed = urlparse(OLLAMA_BASE_URL)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "http://localhost:11434"
    url = f"{base}/api/ps"
    request = urllib.request.Request(url, headers={"User-Agent": "KibaBot/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            return payload.get("models", [])
    except Exception as exc:
        logger.debug("Ollama /api/ps failed: %s", exc)
        return []
```

**Step 4: Run test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/test_models_command.py -v
```
Expected: 2 PASS.

**Step 5: Add `!models` command to `cogs/chat_commands.py`**

```python
@commands.command(name="models")
async def loaded_models(self, ctx):
    """Shows which AI models are currently loaded in VRAM via Ollama."""
    models = await asyncio.to_thread(self.hardware_service.get_ollama_running_models)

    embed = discord.Embed(
        title="🤖 Active AI Models",
        color=discord.Color.dark_blue(),
    )

    if not models:
        embed.description = "No models currently loaded in VRAM."
    else:
        for m in models:
            name = m.get("name", "unknown")
            size_gb = round(m.get("size", 0) / 1e9, 1)
            vram_size = m.get("size_vram", 0)
            vram_gb = round(vram_size / 1e9, 1) if vram_size else "?"
            embed.add_field(
                name=name,
                value=f"Size: {size_gb}GB | VRAM: {vram_gb}GB",
                inline=False,
            )

    await ctx.send(embed=embed)
```

**Step 6: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 7: Commit**

```bash
git add services/hardware_service.py cogs/chat_commands.py tests/test_models_command.py
git commit -m "feat: !models command — shows Ollama models currently loaded in VRAM"
```

---

## Task 10: VRAM OOM notification to owner

**Files:**
- Modify: `tasks/vram_guard.py` — send DM to bot owner on guard trigger
- Test: none needed (integration behavior, hard to unit test cleanly)

**Step 1: Update `guard_loop` in `tasks/vram_guard.py`**

After the stabilization log, add a DM notification to the bot owner:

```python
@tasks.loop(minutes=5)
async def guard_loop(self):
    if getattr(self.bot, "generating_count", 0) > 0:
        return

    current_usage = await asyncio.to_thread(self._get_vram_usage_mb)

    if current_usage > self.vram_threshold_mb:
        logger.info("[VRAM GUARD] High idle usage detected: %sMB. Initializing stabilizer...", current_usage)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        new_usage = await asyncio.to_thread(self._get_vram_usage_mb)
        freed = current_usage - new_usage
        logger.info("[VRAM GUARD] Stabilization complete. Freed %sMB. Current: %sMB.", freed, new_usage)

        # Notify bot owner via DM
        try:
            app_info = await self.bot.application_info()
            owner = app_info.owner
            if owner:
                await owner.send(
                    f"⚠️ **VRAM Guard triggered**\n"
                    f"Idle VRAM was `{current_usage}MB`, freed `{freed}MB`, now `{new_usage}MB`."
                )
        except Exception as exc:
            logger.debug("[VRAM GUARD] Could not DM owner: %s", exc)
```

**Step 2: Commit**

```bash
git add tasks/vram_guard.py
git commit -m "feat: VRAM guard sends DM to bot owner when idle VRAM purge triggers"
```

---

## Task 11: `!purge` command (owner only) — wipe channel history

**Files:**
- Modify: `database/chat_memory.py` — add `delete_channel_history()`
- Modify: `cogs/chat_commands.py` — add `!purge` command
- Test: `tests/test_forget_command.py` (extend existing file)

**Step 1: Add failing test to `tests/test_forget_command.py`**

```python
@pytest.mark.asyncio
async def test_delete_channel_history_runs_without_error():
    from database.chat_memory import delete_channel_history
    await delete_channel_history("888888")
```

**Step 2: Run test to verify it fails**

```
.venv\Scripts\python.exe -m pytest tests/test_forget_command.py::test_delete_channel_history_runs_without_error -v
```
Expected: ImportError.

**Step 3: Add `delete_channel_history` to `database/chat_memory.py`**

```python
async def delete_channel_history(channel_id: str):
    """Wipes ALL chat history for every user in a given channel."""
    db = await get_db()
    await db.execute(
        "DELETE FROM chat_sessions WHERE channel_id = ?",
        (channel_id,),
    )
    await db.execute(
        "DELETE FROM chat_summaries WHERE channel_id = ?",
        (channel_id,),
    )
    await db.execute(
        "DELETE FROM chat_state WHERE channel_id = ?",
        (channel_id,),
    )
    await db.commit()
```

**Step 4: Run test to verify it passes**

```
.venv\Scripts\python.exe -m pytest tests/test_forget_command.py -v
```
Expected: all PASS.

**Step 5: Add `!purge` command to `cogs/chat_commands.py`**

```python
@commands.command(name="purge")
@commands.is_owner()
async def purge_channel(self, ctx):
    """[Owner only] Wipes all chat history for every user in this channel."""
    from database.chat_memory import delete_channel_history
    channel_id = str(ctx.channel.id)
    await delete_channel_history(channel_id)
    embed = discord.Embed(
        title="🗑️ Channel History Purged",
        description=f"All chat history for **#{ctx.channel.name}** has been deleted.",
        color=discord.Color.dark_red(),
    )
    await ctx.send(embed=embed)
```

**Step 6: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 7: Commit**

```bash
git add database/chat_memory.py cogs/chat_commands.py tests/test_forget_command.py
git commit -m "feat: !purge command (owner-only) — wipes all chat history in a channel"
```

---

## Task 12: Persist cooldowns across restarts

**Files:**
- Modify: `cogs/chat_commands.py` — replace in-memory `user_cooldowns` dict with DB-backed version
- Modify: `database/chat_memory.py` — add `get_last_used()` / `set_last_used()`

**Step 1: Add to `database/chat_memory.py`**

First, add the table to `init_chat_memory_db()`:

```python
await db.execute("""
    CREATE TABLE IF NOT EXISTS user_cooldowns (
        user_id TEXT PRIMARY KEY,
        last_used_ts REAL NOT NULL DEFAULT 0
    )
""")
```

Then add functions:

```python
async def get_last_used(user_id: str) -> float:
    db = await get_db()
    cursor = await db.execute(
        "SELECT last_used_ts FROM user_cooldowns WHERE user_id = ?",
        (user_id,),
    )
    row = await cursor.fetchone()
    return row[0] if row else 0.0


async def set_last_used(user_id: str, ts: float):
    db = await get_db()
    await db.execute(
        """INSERT INTO user_cooldowns (user_id, last_used_ts) VALUES (?, ?)
           ON CONFLICT(user_id) DO UPDATE SET last_used_ts = excluded.last_used_ts""",
        (user_id, ts),
    )
    await db.commit()
```

**Step 2: Update `is_on_cooldown` in `cogs/chat_commands.py`**

The current method is synchronous but DB ops are async. Make it async:

```python
async def is_on_cooldown(self, user_id: int, seconds: float = CHAT_COOLDOWN_SECONDS) -> bool:
    from database.chat_memory import get_last_used, set_last_used
    import time
    now = time.time()
    last_used = await get_last_used(str(user_id))
    if now - last_used < seconds:
        return True
    await set_last_used(str(user_id), now)
    return False
```

Update callers in `handle_chat_turn` and `handle_natural_chat` to `await self.is_on_cooldown(...)`.

Remove `self.user_cooldowns = {}` from `__init__`.

**Step 3: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 4: Commit**

```bash
git add database/chat_memory.py cogs/chat_commands.py
git commit -m "feat: persist user cooldowns to DB — survive bot restarts"
```

---

## Task 13: Session info on `!status`

**Files:**
- Modify: `cogs/chat_commands.py` — enhance `kiba_dashboard` embed

**Step 1: Add session count query to `database/chat_memory.py`**

```python
async def get_active_session_count() -> int:
    """Returns number of unique user/channel sessions updated in the last 24 hours."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT COUNT(*) FROM chat_sessions WHERE updated_at >= datetime('now', '-1 day')"
    )
    row = await cursor.fetchone()
    return row[0] if row else 0
```

**Step 2: Update `kiba_dashboard` in `cogs/chat_commands.py`**

After the existing embed fields, add:

```python
from database.chat_memory import get_active_session_count

active_sessions = await get_active_session_count()
embed.add_field(name="Active Sessions (24h)", value=str(active_sessions), inline=True)
```

**Step 3: Commit**

```bash
git add database/chat_memory.py cogs/chat_commands.py
git commit -m "feat: !status shows active session count from last 24 hours"
```

---

## Task 14: Channel allowlist as DB-backed rules

**Files:**
- Modify: `database/chat_memory.py` — add `get_allowed_channels()` / `add_allowed_channel()` / `remove_allowed_channel()`
- Modify: `cogs/chat_commands.py` — add `!allow` / `!deny` commands; load from DB on startup
- Modify: `database/database.py` — add `allowed_channels` table to init

**Step 1: Add table to `init_chat_memory_db()`**

```python
await db.execute("""
    CREATE TABLE IF NOT EXISTS allowed_channels (
        channel_name TEXT PRIMARY KEY
    )
""")
```

Seed it with the current hardcoded list from `core/constants.py` on first boot (check if table is empty).

**Step 2: Add DB functions**

```python
async def get_allowed_channels() -> list[str]:
    db = await get_db()
    cursor = await db.execute("SELECT channel_name FROM allowed_channels")
    rows = await cursor.fetchall()
    return [r[0] for r in rows]

async def add_allowed_channel(channel_name: str):
    db = await get_db()
    await db.execute(
        "INSERT OR IGNORE INTO allowed_channels (channel_name) VALUES (?)",
        (channel_name,),
    )
    await db.commit()

async def remove_allowed_channel(channel_name: str):
    db = await get_db()
    await db.execute(
        "DELETE FROM allowed_channels WHERE channel_name = ?",
        (channel_name,),
    )
    await db.commit()
```

**Step 3: Update `ChatCommands` to load from DB**

In `__init__`, keep `self.allowed_chat_channels` as a set. In `cog_load` (or a new `async_init` method called from setup):

```python
async def cog_load(self):
    from database.chat_memory import get_allowed_channels
    from core.constants import BOT_ALLOWED_CHAT_CHANNELS
    db_channels = await get_allowed_channels()
    if not db_channels:
        # Seed from constants on first run
        from database.chat_memory import add_allowed_channel
        for ch in BOT_ALLOWED_CHAT_CHANNELS:
            await add_allowed_channel(ch)
        self.allowed_chat_channels = set(BOT_ALLOWED_CHAT_CHANNELS)
    else:
        self.allowed_chat_channels = set(db_channels)
```

**Step 4: Add `!allow` / `!deny` commands**

```python
@commands.command(name="allow")
@commands.is_owner()
async def allow_channel(self, ctx, channel_name: str = None):
    """[Owner] Add a channel to Kiba's allowed chat list (no restart needed)."""
    from database.chat_memory import add_allowed_channel
    name = channel_name or ctx.channel.name
    await add_allowed_channel(name.lower())
    self.allowed_chat_channels.add(name.lower())
    await ctx.send(f"✅ `#{name}` added to allowed channels.")

@commands.command(name="deny")
@commands.is_owner()
async def deny_channel(self, ctx, channel_name: str = None):
    """[Owner] Remove a channel from Kiba's allowed chat list."""
    from database.chat_memory import remove_allowed_channel
    name = channel_name or ctx.channel.name
    await remove_allowed_channel(name.lower())
    self.allowed_chat_channels.discard(name.lower())
    await ctx.send(f"✅ `#{name}` removed from allowed channels.")
```

**Step 5: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 6: Commit**

```bash
git add database/chat_memory.py cogs/chat_commands.py
git commit -m "feat: channel allowlist is now DB-backed with !allow and !deny commands (no restart needed)"
```

---

## Task 15: Final cleanup — remove unreachable video debug print

**Files:**
- Modify: `services/video_service.py` — remove unreachable code after return

**Step 1: Open `services/video_service.py` around line 61**

Find the `print("[DEBUG] Video Request..."` line that appears after a `return` statement. Delete it.

**Step 2: Run all tests one final time**

```
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all green.

**Step 3: Final commit**

```bash
git add services/video_service.py
git commit -m "fix: remove unreachable debug print in video_service after return statement"
```

---

## Final Push

```bash
git push origin main
```

---

## Summary of Changes

| Task | Type | File(s) |
|------|------|---------|
| 1 | Perf | `database/db_connection.py` (new), `database/chat_memory.py`, `database/database.py`, `bot.py` |
| 2 | Refactor | `services/llm_service.py` |
| 3 | Perf | `core/executors.py` (new), `services/image_service.py`, `services/music_service.py` |
| 4 | Fix | `services/image_service.py` |
| 5 | Feature | `services/circuit_breaker.py` (new), `services/llm_service.py` |
| 6 | Feature | `services/llm_service.py`, `cogs/chat_commands.py` |
| 7 | Feature | `services/llm_service.py`, `cogs/chat_commands.py` |
| 8 | Feature | `database/chat_memory.py`, `cogs/chat_commands.py` |
| 9 | Feature | `services/hardware_service.py`, `cogs/chat_commands.py` |
| 10 | Feature | `tasks/vram_guard.py` |
| 11 | Feature | `database/chat_memory.py`, `cogs/chat_commands.py` |
| 12 | Feature | `database/chat_memory.py`, `cogs/chat_commands.py` |
| 13 | Feature | `database/chat_memory.py`, `cogs/chat_commands.py` |
| 14 | Feature | `database/chat_memory.py`, `cogs/chat_commands.py` |
| 15 | Fix | `services/video_service.py` |
