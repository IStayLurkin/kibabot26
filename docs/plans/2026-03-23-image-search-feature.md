# Image Search Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When the user explicitly asks to be shown something (e.g. "show me cat memes"), the bot searches Giphy then a local folder, scans every candidate URL with VirusTotal, and posts the first clean result as a Discord file attachment.

**Architecture:** `chat_commands.py` detects explicit image requests before the LLM path, extracts a topic keyword, and calls `image_search_service.py` which tries Giphy then local folder in order. Every candidate URL passes through `virustotal_service.py` before being sent. Flagged results are silently skipped.

**Tech Stack:** Python 3.12, discord.py, aiohttp (async HTTP), Giphy REST API, VirusTotal v3 REST API, pathlib for local folder glob.

---

### Task 1: Add env vars and config

**Files:**
- Modify: `core/config.py`
- Modify: `.env`

**Step 1: Add keys to `.env`**

Open `.env` and append:
```
VIRUSTOTAL_API_KEY=your_vt_key_here
GIPHY_API_KEY=your_giphy_key_here
LOCAL_IMAGE_DIR=G:/path/to/your/local/images
```

**Step 2: Expose them in `core/config.py`**

Add after the `GALLERY_CHANNEL_ID` line (near bottom of file):
```python
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY", "").strip()
LOCAL_IMAGE_DIR = os.getenv("LOCAL_IMAGE_DIR", "").strip()
```

**Step 3: Commit**
```bash
git add core/config.py .env
git commit -m "chore: add image search and virustotal env vars"
```

---

### Task 2: Write `virustotal_service.py`

**Files:**
- Create: `services/virustotal_service.py`
- Create: `tests/test_virustotal_service.py`

**Step 1: Write the failing tests**

Create `tests/test_virustotal_service.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.virustotal_service import is_safe


@pytest.mark.asyncio
async def test_is_safe_returns_true_for_clean_url():
    """Clean URL (0 malicious/suspicious) returns True."""
    mock_analysis = {
        "data": {"attributes": {"stats": {"malicious": 0, "suspicious": 0}}}
    }
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", "fake_key"), \
         patch("services.virustotal_service._submit_url", new_callable=AsyncMock, return_value="scan123"), \
         patch("services.virustotal_service._poll_result", new_callable=AsyncMock, return_value=mock_analysis):
        result = await is_safe("https://media.giphy.com/media/abc/giphy.gif")
    assert result is True


@pytest.mark.asyncio
async def test_is_safe_returns_false_for_malicious_url():
    """Malicious URL returns False."""
    mock_analysis = {
        "data": {"attributes": {"stats": {"malicious": 3, "suspicious": 0}}}
    }
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", "fake_key"), \
         patch("services.virustotal_service._submit_url", new_callable=AsyncMock, return_value="scan123"), \
         patch("services.virustotal_service._poll_result", new_callable=AsyncMock, return_value=mock_analysis):
        result = await is_safe("https://evil.example.com/bad.gif")
    assert result is False


@pytest.mark.asyncio
async def test_is_safe_returns_false_on_exception():
    """Network error returns False (fail safe)."""
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", "fake_key"), \
         patch("services.virustotal_service._submit_url", new_callable=AsyncMock, side_effect=Exception("timeout")):
        result = await is_safe("https://media.giphy.com/media/abc/giphy.gif")
    assert result is False


@pytest.mark.asyncio
async def test_is_safe_returns_false_when_no_api_key():
    """Missing API key returns False immediately."""
    with patch("services.virustotal_service.VIRUSTOTAL_API_KEY", ""):
        result = await is_safe("https://media.giphy.com/media/abc/giphy.gif")
    assert result is False
```

**Step 2: Run tests — verify they fail**
```bash
pytest tests/test_virustotal_service.py -v --timeout=5
```
Expected: `ImportError` or `ModuleNotFoundError` — file doesn't exist yet.

**Step 3: Implement `services/virustotal_service.py`**

```python
import asyncio
import base64
import aiohttp
from core.config import VIRUSTOTAL_API_KEY
from core.logging_config import get_logger

logger = get_logger(__name__)

_VT_BASE = "https://www.virustotal.com/api/v3"
_POLL_ATTEMPTS = 10
_POLL_DELAY = 3


async def _submit_url(session: aiohttp.ClientSession, url: str) -> str:
    """Submit URL to VirusTotal, return analysis ID."""
    headers = {"x-apikey": VIRUSTOTAL_API_KEY, "Content-Type": "application/x-www-form-urlencoded"}
    async with session.post(f"{_VT_BASE}/urls", data=f"url={url}", headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["data"]["id"]


async def _poll_result(session: aiohttp.ClientSession, analysis_id: str) -> dict:
    """Poll analysis until complete, return result dict."""
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    for _ in range(_POLL_ATTEMPTS):
        async with session.get(f"{_VT_BASE}/analyses/{analysis_id}", headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            status = data.get("data", {}).get("attributes", {}).get("status")
            if status == "completed":
                return data
        await asyncio.sleep(_POLL_DELAY)
    return {}


async def is_safe(url: str) -> bool:
    """Return True only if VirusTotal reports 0 malicious and 0 suspicious detections."""
    if not VIRUSTOTAL_API_KEY:
        logger.warning("[virustotal] No API key configured — skipping scan, treating as unsafe")
        return False
    try:
        async with aiohttp.ClientSession() as session:
            analysis_id = await _submit_url(session, url)
            result = await _poll_result(session, analysis_id)
            stats = result.get("data", {}).get("attributes", {}).get("stats", {})
            return stats.get("malicious", 1) == 0 and stats.get("suspicious", 1) == 0
    except Exception:
        logger.exception("[virustotal] Scan failed for %s — treating as unsafe", url)
        return False
```

**Step 4: Run tests — verify they pass**
```bash
pytest tests/test_virustotal_service.py -v --timeout=5
```
Expected: 4 passed.

**Step 5: Commit**
```bash
git add services/virustotal_service.py tests/test_virustotal_service.py
git commit -m "feat: add virustotal URL scanning service"
```

---

### Task 3: Write `image_search_service.py`

**Files:**
- Create: `services/image_search_service.py`
- Create: `tests/test_image_search_service.py`

**Step 1: Write the failing tests**

Create `tests/test_image_search_service.py`:
```python
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from services.image_search_service import search_giphy, search_local, find_verified_image


@pytest.mark.asyncio
async def test_search_giphy_returns_urls():
    """Giphy search returns list of gif URLs."""
    mock_response = {
        "data": [
            {"images": {"original": {"url": "https://media.giphy.com/media/abc/giphy.gif"}}},
            {"images": {"original": {"url": "https://media.giphy.com/media/def/giphy.gif"}}},
        ]
    }
    with patch("services.image_search_service.GIPHY_API_KEY", "fake_key"), \
         patch("services.image_search_service._giphy_get", new_callable=AsyncMock, return_value=mock_response):
        result = await search_giphy("cat memes")
    assert len(result) == 2
    assert result[0] == "https://media.giphy.com/media/abc/giphy.gif"


@pytest.mark.asyncio
async def test_search_giphy_returns_empty_on_error():
    """Giphy error returns empty list."""
    with patch("services.image_search_service.GIPHY_API_KEY", "fake_key"), \
         patch("services.image_search_service._giphy_get", new_callable=AsyncMock, side_effect=Exception("timeout")):
        result = await search_giphy("cat memes")
    assert result == []


def test_search_local_matches_by_keyword(tmp_path):
    """Local search finds files whose name contains the keyword."""
    (tmp_path / "cat_funny.gif").touch()
    (tmp_path / "dog_photo.jpg").touch()
    (tmp_path / "cat_meme.png").touch()

    with patch("services.image_search_service.LOCAL_IMAGE_DIR", str(tmp_path)):
        result = search_local("cat")
    assert len(result) == 2
    assert all("cat" in Path(p).name for p in result)


def test_search_local_returns_empty_when_no_dir():
    """Local search returns empty list when LOCAL_IMAGE_DIR is not set."""
    with patch("services.image_search_service.LOCAL_IMAGE_DIR", ""):
        result = search_local("cat")
    assert result == []


@pytest.mark.asyncio
async def test_find_verified_image_returns_first_safe():
    """find_verified_image returns first URL that passes VT scan."""
    candidates = ["https://media.giphy.com/media/abc/giphy.gif", "https://media.giphy.com/media/def/giphy.gif"]
    with patch("services.image_search_service.search_giphy", new_callable=AsyncMock, return_value=candidates), \
         patch("services.image_search_service.search_local", return_value=[]), \
         patch("services.image_search_service.is_safe", new_callable=AsyncMock, return_value=True):
        result = await find_verified_image("cat memes")
    assert result == candidates[0]


@pytest.mark.asyncio
async def test_find_verified_image_skips_unsafe():
    """find_verified_image skips unsafe URLs and returns next safe one."""
    candidates = ["https://evil.example.com/bad.gif", "https://media.giphy.com/media/safe/giphy.gif"]
    with patch("services.image_search_service.search_giphy", new_callable=AsyncMock, return_value=candidates), \
         patch("services.image_search_service.search_local", return_value=[]), \
         patch("services.image_search_service.is_safe", new_callable=AsyncMock, side_effect=[False, True]):
        result = await find_verified_image("cat")
    assert result == "https://media.giphy.com/media/safe/giphy.gif"


@pytest.mark.asyncio
async def test_find_verified_image_returns_none_when_all_unsafe():
    """Returns None when all candidates fail VT scan."""
    candidates = ["https://bad1.com/a.gif", "https://bad2.com/b.gif"]
    with patch("services.image_search_service.search_giphy", new_callable=AsyncMock, return_value=candidates), \
         patch("services.image_search_service.search_local", return_value=[]), \
         patch("services.image_search_service.is_safe", new_callable=AsyncMock, return_value=False):
        result = await find_verified_image("cat")
    assert result is None
```

**Step 2: Run tests — verify they fail**
```bash
pytest tests/test_image_search_service.py -v --timeout=5
```
Expected: `ImportError` — file doesn't exist yet.

**Step 3: Implement `services/image_search_service.py`**

```python
import aiohttp
from pathlib import Path
from core.config import GIPHY_API_KEY, LOCAL_IMAGE_DIR
from core.logging_config import get_logger
from services.virustotal_service import is_safe

logger = get_logger(__name__)

_GIPHY_SEARCH = "https://api.giphy.com/v1/gifs/search"
_SUPPORTED_EXTS = {".gif", ".png", ".jpg", ".jpeg", ".webp"}
_MAX_CANDIDATES = 5


async def _giphy_get(topic: str) -> dict:
    params = {"api_key": GIPHY_API_KEY, "q": topic, "limit": _MAX_CANDIDATES, "rating": "pg-13"}
    async with aiohttp.ClientSession() as session:
        async with session.get(_GIPHY_SEARCH, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()


async def search_giphy(topic: str) -> list[str]:
    """Return up to 5 Giphy GIF URLs for the topic."""
    if not GIPHY_API_KEY:
        return []
    try:
        data = await _giphy_get(topic)
        return [
            item["images"]["original"]["url"]
            for item in data.get("data", [])
            if item.get("images", {}).get("original", {}).get("url")
        ]
    except Exception:
        logger.exception("[image_search] Giphy search failed for %r", topic)
        return []


def search_local(topic: str) -> list[str]:
    """Return local image paths whose filename contains the topic keyword."""
    if not LOCAL_IMAGE_DIR:
        return []
    folder = Path(LOCAL_IMAGE_DIR)
    if not folder.is_dir():
        return []
    keyword = topic.lower().replace(" ", "_")
    results = []
    for ext in _SUPPORTED_EXTS:
        results.extend(
            str(p) for p in folder.iterdir()
            if p.suffix.lower() == ext and keyword in p.name.lower()
        )
    return results[:_MAX_CANDIDATES]


async def find_verified_image(topic: str) -> str | None:
    """
    Search Giphy then local folder for topic. Return first URL/path
    that passes VirusTotal scan, or None if all fail.
    """
    candidates = await search_giphy(topic)

    # Local files don't have URLs to scan — convert to file:// for VT or just trust them?
    # We scan local files by their path string; VT won't scan local paths so we trust them
    # but still run through the pipeline for uniformity — is_safe will fail-safe to False
    # for non-http paths. So we append local paths after URL candidates.
    local = search_local(topic)

    for url in candidates:
        if await is_safe(url):
            return url

    # For local files, bypass VT (they're on your machine — you put them there)
    if local:
        return local[0]

    logger.info("[image_search] No safe results found for %r", topic)
    return None
```

**Step 4: Run tests — verify they pass**
```bash
pytest tests/test_image_search_service.py -v --timeout=5
```
Expected: 6 passed.

**Step 5: Commit**
```bash
git add services/image_search_service.py tests/test_image_search_service.py
git commit -m "feat: add image search service (Giphy + local folder + VT scanning)"
```

---

### Task 4: Add image request detection to `chat_router.py`

**Files:**
- Modify: `services/chat_router.py`
- Create: `tests/test_chat_router_image.py`

**Step 1: Write the failing tests**

Create `tests/test_chat_router_image.py`:
```python
from services.chat_router import extract_image_request


def test_detects_show_me_memes():
    assert extract_image_request("show me cat memes") == "cat memes"

def test_detects_send_me():
    assert extract_image_request("send me a cooking guide") == "cooking guide"

def test_detects_got_any():
    assert extract_image_request("got any dog gifs") == "dog gifs"

def test_detects_find_me():
    assert extract_image_request("find me funny cat pics") == "funny cat pics"

def test_detects_post():
    assert extract_image_request("post some memes") == "some memes"

def test_no_match_returns_none():
    assert extract_image_request("how ya feeling") is None

def test_no_match_plain_chat():
    assert extract_image_request("what time is it") is None

def test_detects_share():
    assert extract_image_request("share a cat meme with me") == "cat meme"
```

**Step 2: Run tests — verify they fail**
```bash
pytest tests/test_chat_router_image.py -v --timeout=5
```
Expected: `ImportError: cannot import name 'extract_image_request'`

**Step 3: Add `extract_image_request` to `services/chat_router.py`**

Append to the bottom of `services/chat_router.py`:
```python
_IMAGE_REQUEST = re.compile(
    r"(?:show|send|post|find|got|share)\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?(.+?)(?:\s+with me)?$",
    re.IGNORECASE,
)

_MEDIA_KEYWORDS = re.compile(
    r"\b(?:meme|memes|gif|gifs|pic|pics|image|images|photo|photos|video|videos|guide|tutorial)\b",
    re.IGNORECASE,
)


def extract_image_request(text: str) -> str | None:
    """
    If the message is an explicit request to show/send/post media,
    return the topic keyword string. Otherwise return None.
    """
    m = _IMAGE_REQUEST.match(text.strip())
    if not m:
        return None
    topic = m.group(1).strip()
    # Must contain a media keyword to avoid false positives like "show me how to cook"
    if not _MEDIA_KEYWORDS.search(topic):
        return None
    return topic
```

**Step 4: Run tests — verify they pass**
```bash
pytest tests/test_chat_router_image.py -v --timeout=5
```
Expected: 8 passed.

**Step 5: Commit**
```bash
git add services/chat_router.py tests/test_chat_router_image.py
git commit -m "feat: add extract_image_request detection to chat_router"
```

---

### Task 5: Wire into `chat_commands.py`

**Files:**
- Modify: `cogs/chat_commands.py`

**Step 1: Add imports at top of `cogs/chat_commands.py`**

After the existing imports block, add:
```python
from services.chat_router import extract_image_request
from services.image_search_service import find_verified_image
```

**Step 2: Add the image handling block in `handle_chat_turn`**

In `handle_chat_turn`, inside the `async with destination.typing():` block, add a new branch **before** the existing `intent = self.dispatcher.classify_intent(content)` line:

```python
                # Image search path — check before LLM
                image_topic = extract_image_request(content)
                if image_topic:
                    url_or_path = await find_verified_image(image_topic)
                    if url_or_path:
                        from pathlib import Path as _Path
                        p = _Path(url_or_path)
                        if p.exists():
                            # Local file
                            await destination.send(file=discord.File(str(p), filename=p.name))
                        else:
                            # Remote URL — download and send as attachment
                            import aiohttp as _aiohttp
                            async with _aiohttp.ClientSession() as _sess:
                                async with _sess.get(url_or_path) as _resp:
                                    if _resp.status == 200:
                                        data = await _resp.read()
                                        ext = _Path(url_or_path.split("?")[0]).suffix or ".gif"
                                        fname = f"kiba_image{ext}"
                                        await destination.send(file=discord.File(
                                            fp=__import__("io").BytesIO(data),
                                            filename=fname
                                        ))
                    return
```

**Step 3: Restart bot and test manually**

Send in Discord:
- `show me cat memes` — should post a GIF
- `show me dog pics` — should post a GIF
- `how ya feeling` — should go through normal LLM path (not image search)

**Step 4: Commit**
```bash
git add cogs/chat_commands.py
git commit -m "feat: wire image search into chat pipeline"
```

---

### Task 6: Update `_URL_SENTENCE` stripping to not strip local file sends

This is already fine — the URL stripping in `llm_service.py` only applies to LLM text output. The image pipeline returns before the LLM path so there's nothing to strip. No change needed.

---

### Task 7: Push to GitHub

```bash
git push
```

---

## Notes

- **VirusTotal free tier**: 4 requests/minute, 500/day. For a personal bot this is plenty.
- **Giphy free tier**: 100 req/hour. Fine for personal use.
- **Local files**: trusted without VT scan (you put them there). Remote Giphy URLs are always scanned.
- **aiohttp** must be installed: `pip install aiohttp` (likely already present — check requirements.txt).
