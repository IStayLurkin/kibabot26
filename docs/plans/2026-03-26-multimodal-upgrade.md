# Multimodal Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add vision analysis, thinking/reasoning models, tiered coding models, Fish Speech TTS, Parakeet STT, and Mem0 memory comparison — all integrated into existing commands with fast/best tiers.

**Architecture:** Four new services (VisionService, ThinkingService, FishSpeechService, ParakeetService, Mem0Service) wired into existing cogs. All Ollama-based services (vision, thinking, coding) reuse `LLMService._create_chat_completion`. Audio services are subprocess-based like existing Piper/Whisper. Memory backend is swappable via DB config flag.

**Tech Stack:** Ollama (multimodal API with `images` field), Fish Speech V1.5 (subprocess), NVIDIA NeMo Parakeet V3 (subprocess), mem0ai Python client, discord.py attachments API.

---

## Phase 1 — Thinking & Coding Models (zero new infrastructure)

### Task 1: Config — thinking and coding model names

**Files:**
- Modify: `core/config.py`

**Step 1: Add config entries after the existing OLLAMA_NUM_CTX line**

```python
# --- THINKING MODELS ---
THINKING_FAST_MODEL = os.getenv("THINKING_FAST_MODEL", "deepseek-r1:7b").strip()
THINKING_BEST_MODEL = os.getenv("THINKING_BEST_MODEL", "deepseek-r1:32b").strip()

# --- CODING MODELS ---
CODING_FAST_MODEL = os.getenv("CODING_FAST_MODEL", "qwen2.5-coder:7b").strip()
CODING_BEST_MODEL = os.getenv("CODING_BEST_MODEL", "devstral:24b").strip()
```

**Step 2: Verify config loads**

Run: `cd G:/code/python/learn_python/bot/discord_bot_things && .venv/Scripts/python.exe -c "from core.config import THINKING_FAST_MODEL, THINKING_BEST_MODEL, CODING_FAST_MODEL, CODING_BEST_MODEL; print(THINKING_FAST_MODEL, THINKING_BEST_MODEL, CODING_FAST_MODEL, CODING_BEST_MODEL)"`
Expected: `deepseek-r1:7b deepseek-r1:32b qwen2.5-coder:7b devstral:24b`

**Step 3: Commit**

```bash
git add core/config.py
git commit -m "feat: add thinking and coding model config entries"
```

---

### Task 2: ThinkingService

**Files:**
- Create: `services/thinking_service.py`

**Step 1: Write the service**

```python
from __future__ import annotations

import asyncio
from core.config import (
    OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_NUM_CTX,
    THINKING_FAST_MODEL, THINKING_BEST_MODEL,
)
from core.logging_config import get_logger
from openai import OpenAI

logger = get_logger(__name__)

THINKING_TIERS = {
    "fast": THINKING_FAST_MODEL,
    "best": THINKING_BEST_MODEL,
}


class ThinkingService:
    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    def _think_sync(self, prompt: str, tier: str) -> str:
        model = THINKING_TIERS.get(tier, THINKING_FAST_MODEL)
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=4096,
            extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX * 2}},
        )
        return response.choices[0].message.content or ""

    async def think(self, prompt: str, tier: str = "fast") -> str:
        """Run a thinking/reasoning model. Returns final answer with <think> blocks stripped."""
        import re
        raw = await asyncio.to_thread(self._think_sync, prompt, tier)
        # Strip <think>...</think> blocks — already handled by _sanitize_model_text but do it here too
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
        return cleaned if cleaned else raw.strip()
```

**Step 2: Verify import**

Run: `.venv/Scripts/python.exe -c "from services.thinking_service import ThinkingService; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add services/thinking_service.py
git commit -m "feat: add ThinkingService wrapping deepseek-r1 fast/best tiers"
```

---

### Task 3: !think command cog

**Files:**
- Create: `cogs/thinking_commands.py`
- Modify: `bot.py`

**Step 1: Write the cog**

```python
from __future__ import annotations

import discord
from discord.ext import commands

from core.config import THINKING_FAST_MODEL, THINKING_BEST_MODEL
from core.logging_config import get_logger
from services.thinking_service import THINKING_TIERS

logger = get_logger(__name__)

MAX_THINK_LENGTH = 3800


class ThinkingCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.group(name="think", invoke_without_command=True, help="Run a reasoning/thinking model on a prompt.")
    async def think_group(self, ctx: commands.Context, *, prompt: str = ""):
        if not prompt:
            await ctx.send(
                f"Usage: `!think fast <prompt>` or `!think best <prompt>`\n"
                f"fast → `{THINKING_FAST_MODEL}` | best → `{THINKING_BEST_MODEL}`"
            )
            return
        # Default to fast tier if no subcommand
        await self._run_think(ctx, prompt, "fast")

    @think_group.command(name="fast", help=f"Think with {THINKING_FAST_MODEL} (faster).")
    async def think_fast(self, ctx: commands.Context, *, prompt: str):
        await self._run_think(ctx, prompt, "fast")

    @think_group.command(name="best", help=f"Think with {THINKING_BEST_MODEL} (deeper reasoning).")
    async def think_best(self, ctx: commands.Context, *, prompt: str):
        await self._run_think(ctx, prompt, "best")

    async def _run_think(self, ctx: commands.Context, prompt: str, tier: str):
        service = getattr(self.bot, "thinking_service", None)
        if service is None:
            await ctx.send("Thinking service is not available.")
            return
        if len(prompt) > MAX_THINK_LENGTH:
            await ctx.send(f"Prompt too long. Keep it under {MAX_THINK_LENGTH} characters.")
            return
        model = THINKING_TIERS.get(tier, THINKING_FAST_MODEL)
        async with ctx.typing():
            try:
                result = await service.think(prompt, tier=tier)
                if not result:
                    await ctx.send("No response from thinking model.")
                    return
                # Send in chunks if long
                limit = 1900
                if len(result) <= limit:
                    await ctx.send(result)
                else:
                    for i in range(0, len(result), limit):
                        await ctx.send(result[i:i+limit])
            except Exception as exc:
                logger.error("[think] Error: %s", exc)
                await ctx.send(f"Thinking model error: {exc}")


async def setup(bot: commands.Bot):
    await bot.add_cog(ThinkingCommands(bot))
```

**Step 2: Add ThinkingService to bot.py**

In `bot.py`, add import:
```python
from services.thinking_service import ThinkingService
```

In `ExpenseBot.__init__`, add:
```python
self.thinking_service = None
```

In `setup_hook`, after `self.wan_service = WanService(...)`:
```python
self.thinking_service = ThinkingService(performance_tracker=self.performance_tracker)
```

Add to extensions list:
```python
"cogs.thinking_commands",
```

**Step 3: Verify bot loads**

Run: `.venv/Scripts/python.exe -c "import bot; print('OK')"` (will fail if syntax error)
Expected: No import error

**Step 4: Commit**

```bash
git add cogs/thinking_commands.py bot.py
git commit -m "feat: add !think fast/best commands via ThinkingService"
```

---

### Task 4: !code ask subcommand

**Files:**
- Modify: `cogs/code_commands.py`
- Modify: `services/codegen_service.py` (add ask method if not present)

**Step 1: Check codegen_service.py for existing ask/generate methods**

Read `services/codegen_service.py` — look for `generate_code_help` or similar.

**Step 2: Add `ask` method to CodegenService if needed**

In `services/codegen_service.py`, add:
```python
from core.config import CODING_FAST_MODEL, CODING_BEST_MODEL

CODING_TIERS = {
    "fast": CODING_FAST_MODEL,
    "best": CODING_BEST_MODEL,
}

async def ask(self, prompt: str, tier: str = "fast") -> str:
    """Ask a coding-specific model a question. Uses tiered model selection."""
    model = CODING_TIERS.get(tier, CODING_FAST_MODEL)
    # Temporarily override model for this call
    original_model = None
    if self.llm_service is not None:
        original_model = self.llm_service._get_active_model_name()
    result = await self.generate_code_help(prompt, model_override=model)
    return result
```

**Step 3: Add `!code ask` subcommand to code_commands.py**

Add after existing `code_group` commands:
```python
@code_group.group(name="ask", invoke_without_command=True, help="Ask a coding model a question.")
async def code_ask_group(self, ctx: commands.Context, *, prompt: str = ""):
    if not prompt:
        await ctx.send("Usage: `!code ask fast <prompt>` or `!code ask best <prompt>`")
        return
    await self._run_code_ask(ctx, prompt, "fast")

@code_ask_group.command(name="fast", help="Ask the fast coding model.")
async def code_ask_fast(self, ctx: commands.Context, *, prompt: str):
    await self._run_code_ask(ctx, prompt, "fast")

@code_ask_group.command(name="best", help="Ask the best coding model.")
async def code_ask_best(self, ctx: commands.Context, *, prompt: str):
    await self._run_code_ask(ctx, prompt, "best")

async def _run_code_ask(self, ctx: commands.Context, prompt: str, tier: str):
    codegen = getattr(self.bot, "codegen_service", None)
    if codegen is None:
        await ctx.send("Codegen service is not available.")
        return
    async with ctx.typing():
        try:
            result = await codegen.ask(prompt, tier=tier)
            if len(result) <= 1900:
                await ctx.send(result)
            else:
                for i in range(0, len(result), 1900):
                    await ctx.send(result[i:i+1900])
        except Exception as exc:
            await ctx.send(f"Code ask error: {exc}")
```

**Step 4: Commit**

```bash
git add cogs/code_commands.py services/codegen_service.py
git commit -m "feat: add !code ask fast/best subcommands for tiered coding models"
```

---

## Phase 2 — Vision

### Task 5: Vision config

**Files:**
- Modify: `core/config.py`

**Step 1: Add vision config after CODING_BEST_MODEL**

```python
# --- VISION MODELS ---
VISION_FAST_MODEL = os.getenv("VISION_FAST_MODEL", "moondream").strip()
VISION_BEST_MODEL = os.getenv("VISION_BEST_MODEL", "llava:34b").strip()
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
```

Note: Use `moondream` (fastest, 2B) as fast tier and `llava:34b` Q4 as best. User can override via .env.

**Step 2: Verify**

Run: `.venv/Scripts/python.exe -c "from core.config import VISION_FAST_MODEL, VISION_BEST_MODEL; print(VISION_FAST_MODEL, VISION_BEST_MODEL)"`
Expected: `moondream llava:34b`

**Step 3: Commit**

```bash
git add core/config.py
git commit -m "feat: add vision model config entries"
```

---

### Task 6: VisionService

**Files:**
- Create: `services/vision_service.py`

**Step 1: Write the service**

```python
from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Union

import httpx

from core.config import (
    OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_NUM_CTX,
    VISION_FAST_MODEL, VISION_BEST_MODEL,
)
from core.logging_config import get_logger
from openai import OpenAI

logger = get_logger(__name__)

VISION_TIERS = {
    "fast": VISION_FAST_MODEL,
    "best": VISION_BEST_MODEL,
}


class VisionService:
    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    async def _fetch_image_b64(self, url: str) -> str:
        """Download image from URL and return base64 string."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")

    def _analyze_sync(self, image_b64: str, prompt: str, model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "Describe this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            max_tokens=1024,
            extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
        )
        return response.choices[0].message.content or ""

    async def analyze_url(self, url: str, prompt: str = "", tier: str = "fast") -> str:
        """Download image from URL and analyze it."""
        model = VISION_TIERS.get(tier, VISION_FAST_MODEL)
        image_b64 = await self._fetch_image_b64(url)
        return await asyncio.to_thread(self._analyze_sync, image_b64, prompt, model)

    async def analyze_bytes(self, image_bytes: bytes, prompt: str = "", tier: str = "fast") -> str:
        """Analyze image from raw bytes."""
        model = VISION_TIERS.get(tier, VISION_FAST_MODEL)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return await asyncio.to_thread(self._analyze_sync, image_b64, prompt, model)
```

**Step 2: Verify import**

Run: `.venv/Scripts/python.exe -c "from services.vision_service import VisionService; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add services/vision_service.py
git commit -m "feat: add VisionService with fast/best tier Ollama multimodal calls"
```

---

### Task 7: !vision command cog + auto-detect in chat

**Files:**
- Create: `cogs/vision_commands.py`
- Modify: `bot.py`
- Modify: `cogs/chat_commands.py` (auto-detect image attachments)

**Step 1: Write vision_commands.py**

```python
from __future__ import annotations

from discord.ext import commands

from core.config import VISION_FAST_MODEL, VISION_BEST_MODEL
from core.logging_config import get_logger
from services.vision_service import VISION_TIERS

logger = get_logger(__name__)


class VisionCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.group(name="vision", invoke_without_command=True, help="Analyze an image with a vision model.")
    async def vision_group(self, ctx: commands.Context, *, prompt: str = ""):
        await ctx.send(
            f"Usage: `!vision fast [prompt]` or `!vision best [prompt]` — attach an image or include a URL.\n"
            f"fast → `{VISION_FAST_MODEL}` | best → `{VISION_BEST_MODEL}`"
        )

    @vision_group.command(name="fast", help="Analyze with fast vision model.")
    async def vision_fast(self, ctx: commands.Context, *, prompt: str = ""):
        await self._run_vision(ctx, prompt, "fast")

    @vision_group.command(name="best", help="Analyze with best vision model.")
    async def vision_best(self, ctx: commands.Context, *, prompt: str = ""):
        await self._run_vision(ctx, prompt, "best")

    async def _run_vision(self, ctx: commands.Context, prompt: str, tier: str):
        service = getattr(self.bot, "vision_service", None)
        if service is None:
            await ctx.send("Vision service is not available.")
            return

        # Get image from attachment or URL in prompt
        image_bytes = None
        image_url = None

        if ctx.message.attachments:
            att = ctx.message.attachments[0]
            if att.content_type and att.content_type.startswith("image/"):
                image_bytes = await att.read()

        if image_bytes is None:
            # Try to find URL in prompt
            import re
            url_match = re.search(r"https?://\S+\.(?:png|jpg|jpeg|gif|webp)", prompt, re.IGNORECASE)
            if url_match:
                image_url = url_match.group(0)
                prompt = prompt.replace(image_url, "").strip()

        if image_bytes is None and image_url is None:
            await ctx.send("Attach an image or include an image URL.")
            return

        async with ctx.typing():
            try:
                if image_bytes is not None:
                    result = await service.analyze_bytes(image_bytes, prompt=prompt, tier=tier)
                else:
                    result = await service.analyze_url(image_url, prompt=prompt, tier=tier)

                if not result:
                    await ctx.send("Vision model returned no response.")
                    return
                if len(result) <= 1900:
                    await ctx.send(result)
                else:
                    for i in range(0, len(result), 1900):
                        await ctx.send(result[i:i+1900])
            except Exception as exc:
                logger.error("[vision] Error: %s", exc)
                await ctx.send(f"Vision error: {exc}")


async def setup(bot: commands.Bot):
    await bot.add_cog(VisionCommands(bot))
```

**Step 2: Wire VisionService into bot.py**

Add import:
```python
from services.vision_service import VisionService
```

In `__init__`:
```python
self.vision_service = None
```

In `setup_hook` after `self.thinking_service = ...`:
```python
self.vision_service = VisionService(performance_tracker=self.performance_tracker)
```

Add to extensions:
```python
"cogs.vision_commands",
```

**Step 3: Auto-detect images in chat (chat_commands.py)**

In `cogs/chat_commands.py`, find the `handle_natural_chat` method. Before calling `generate_dynamic_reply`, add:

```python
# Auto-detect image attachments — analyze with vision service before LLM reply
vision_context = ""
vision_service = getattr(self.bot, "vision_service", None)
if vision_service and message.attachments:
    for att in message.attachments:
        if att.content_type and att.content_type.startswith("image/"):
            try:
                image_bytes = await att.read()
                vision_context = await vision_service.analyze_bytes(
                    image_bytes,
                    prompt=user_text,
                    tier="fast",
                )
                logger.info("[vision_auto] Analyzed attachment for user=%s", message.author.id)
            except Exception as exc:
                logger.warning("[vision_auto] Failed to analyze attachment: %s", exc)
            break  # Only analyze first image

# Prepend vision context to user text if we got a description
if vision_context:
    user_text = f"[Image attached — vision model says: {vision_context}]\n{user_text}"
```

**Step 4: Commit**

```bash
git add cogs/vision_commands.py bot.py cogs/chat_commands.py
git commit -m "feat: add !vision fast/best commands and auto-detect image attachments in chat"
```

---

## Phase 3 — Audio Upgrade

### Task 8: Fish Speech TTS service

**Files:**
- Create: `services/fish_speech_service.py`

**Prerequisites:** Fish Speech V1.5 must be installed. Install via:
```
pip install fish-speech
# or clone from https://github.com/fishaudio/fish-speech and run inference server
```

Fish Speech runs as a local inference server on port 8080 (or configurable). The service calls it via HTTP.

**Step 1: Add Fish Speech config to core/config.py**

```python
# --- FISH SPEECH TTS ---
FISH_SPEECH_BASE_URL = os.getenv("FISH_SPEECH_BASE_URL", "http://localhost:8080").strip()
FISH_SPEECH_ENABLED = os.getenv("FISH_SPEECH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
```

**Step 2: Write the service**

```python
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import httpx

from core.config import FISH_SPEECH_BASE_URL, MEDIA_OUTPUT_DIR
from core.logging_config import get_logger

logger = get_logger(__name__)


class FishSpeechService:
    """TTS via Fish Speech V1.5 local inference server."""

    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
        self.output_dir = Path(MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = FISH_SPEECH_BASE_URL

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/v1/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def synthesize(self, text: str, voice_id: str = "default") -> str | None:
        """Synthesize text to speech. Returns path to WAV file or None on failure."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/tts",
                    json={"text": text, "voice": voice_id, "format": "wav"},
                )
                resp.raise_for_status()
                out_path = self.output_dir / f"fish_{uuid.uuid4().hex[:8]}.wav"
                out_path.write_bytes(resp.content)
                logger.info("[fish_speech] Synthesized %d chars → %s", len(text), out_path)
                return str(out_path)
        except Exception as exc:
            logger.error("[fish_speech] Synthesis failed: %s", exc)
            return None
```

**Step 3: Commit**

```bash
git add services/fish_speech_service.py core/config.py
git commit -m "feat: add FishSpeechService for best-tier TTS"
```

---

### Task 9: Parakeet STT service

**Files:**
- Create: `services/parakeet_service.py`

**Prerequisites:** NVIDIA NeMo Parakeet requires `nemo_toolkit[asr]`. Install:
```
pip install nemo_toolkit[asr]
```

**Step 1: Add Parakeet config to core/config.py**

```python
# --- PARAKEET STT ---
PARAKEET_MODEL = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-1.1b").strip()
PARAKEET_ENABLED = os.getenv("PARAKEET_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
```

**Step 2: Write the service**

```python
from __future__ import annotations

import asyncio
from pathlib import Path

from core.config import PARAKEET_MODEL
from core.logging_config import get_logger

logger = get_logger(__name__)

_nemo_model = None
_nemo_lock = asyncio.Lock()


async def _get_nemo_model():
    global _nemo_model
    async with _nemo_lock:
        if _nemo_model is None:
            import nemo.collections.asr as nemo_asr
            _nemo_model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL)
            _nemo_model.eval()
            logger.info("[parakeet] Model loaded: %s", PARAKEET_MODEL)
    return _nemo_model


class ParakeetService:
    """STT via NVIDIA NeMo Parakeet — faster and more accurate than Whisper base."""

    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file. Returns transcript string."""
        try:
            model = await _get_nemo_model()
            def _run():
                result = model.transcribe([audio_path])
                # result is list of strings or list of Hypothesis objects
                if result and hasattr(result[0], "text"):
                    return result[0].text
                return str(result[0]) if result else ""
            transcript = await asyncio.to_thread(_run)
            logger.info("[parakeet] Transcribed %s → %r", audio_path, transcript[:80])
            return transcript
        except Exception as exc:
            logger.error("[parakeet] Transcription failed: %s", exc)
            return ""
```

**Step 3: Commit**

```bash
git add services/parakeet_service.py core/config.py
git commit -m "feat: add ParakeetService for best-tier STT"
```

---

### Task 10: Upgrade !tts and add !stt toggle

**Files:**
- Modify: `cogs/media_commands.py`
- Modify: `bot.py`

**Step 1: Wire services into bot.py**

Add imports:
```python
from services.fish_speech_service import FishSpeechService
from services.parakeet_service import ParakeetService
```

In `__init__`:
```python
self.fish_speech_service = None
self.parakeet_service = None
```

In `setup_hook`:
```python
from core.config import FISH_SPEECH_ENABLED, PARAKEET_ENABLED
if FISH_SPEECH_ENABLED:
    self.fish_speech_service = FishSpeechService(performance_tracker=self.performance_tracker)
if PARAKEET_ENABLED:
    self.parakeet_service = ParakeetService(performance_tracker=self.performance_tracker)
```

**Step 2: Upgrade !tts in media_commands.py**

Change existing `!tts` command to a group:

```python
@commands.group(name="tts", aliases=["say"], invoke_without_command=True, help="Text-to-speech.")
async def tts_group(self, ctx: commands.Context, *, text: str = ""):
    if not text:
        await ctx.send("Usage: `!tts fast <text>` (Piper) or `!tts best <text>` (Fish Speech)")
        return
    # Default to fast
    await self._run_tts(ctx, text, "fast")

@tts_group.command(name="fast", help="TTS via Piper (fast, local).")
async def tts_fast(self, ctx: commands.Context, *, text: str):
    await self._run_tts(ctx, text, "fast")

@tts_group.command(name="best", help="TTS via Fish Speech V1.5 (expressive).")
async def tts_best(self, ctx: commands.Context, *, text: str):
    await self._run_tts(ctx, text, "best")

async def _run_tts(self, ctx: commands.Context, text: str, tier: str):
    from core.config import MAX_TTS_LENGTH
    if len(text) > MAX_TTS_LENGTH:
        await ctx.send(f"Text too long. Keep it under {MAX_TTS_LENGTH} characters.")
        return
    async with ctx.typing():
        if tier == "best":
            fish = getattr(self.bot, "fish_speech_service", None)
            if fish is None:
                await ctx.send("Fish Speech is not enabled. Set FISH_SPEECH_ENABLED=true in .env.")
                return
            path = await fish.synthesize(text)
        else:
            voice = getattr(self.bot, "voice_service", None)
            if voice is None:
                await ctx.send("Voice service not available.")
                return
            path = await voice.text_to_speech(text)

        if path:
            await ctx.send(file=discord.File(path))
        else:
            await ctx.send("TTS failed.")
```

**Step 3: Add !stt toggle command**

```python
@commands.group(name="stt", invoke_without_command=True, help="Speech-to-text tier settings.")
async def stt_group(self, ctx: commands.Context):
    from database.behavior_rules_repository import get_bot_config
    tier = await get_bot_config("stt_tier", "fast")
    await ctx.send(f"Current STT tier: `{tier}` (fast=Whisper, best=Parakeet). Use `!stt fast` or `!stt best` to switch.")

@stt_group.command(name="fast", help="Use Faster-Whisper for STT.")
async def stt_fast(self, ctx: commands.Context):
    from database.behavior_rules_repository import set_bot_config
    await set_bot_config("stt_tier", "fast")
    await ctx.send("STT set to fast (Faster-Whisper).")

@stt_group.command(name="best", help="Use Parakeet V3 for STT.")
async def stt_best(self, ctx: commands.Context):
    from database.behavior_rules_repository import set_bot_config
    parakeet = getattr(self.bot, "parakeet_service", None)
    if parakeet is None:
        await ctx.send("Parakeet is not enabled. Set PARAKEET_ENABLED=true in .env.")
        return
    await set_bot_config("stt_tier", "best")
    await ctx.send("STT set to best (Parakeet V3).")
```

**Step 4: Update chat_commands.py voice attachment handling**

Find where voice attachments are processed and add tier routing:
```python
from database.behavior_rules_repository import get_bot_config
stt_tier = await get_bot_config("stt_tier", "fast")
if stt_tier == "best":
    parakeet = getattr(self.bot, "parakeet_service", None)
    if parakeet:
        transcript = await parakeet.transcribe(audio_path)
    else:
        transcript = await voice_service.speech_to_text(audio_path)
else:
    transcript = await voice_service.speech_to_text(audio_path)
```

**Step 5: Commit**

```bash
git add cogs/media_commands.py bot.py cogs/chat_commands.py
git commit -m "feat: tier !tts fast/best (Piper/Fish Speech) and !stt fast/best toggle (Whisper/Parakeet)"
```

---

## Phase 4 — Memory Comparison (Mem0)

### Task 11: Mem0Service

**Files:**
- Create: `services/mem0_service.py`

**Prerequisites:**
```
pip install mem0ai
```

**Step 1: Add Mem0 config to core/config.py**

```python
# --- MEM0 MEMORY ---
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "").strip()  # Leave empty for local mode
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
```

**Step 2: Write the service**

```python
from __future__ import annotations

import asyncio
from core.config import MEM0_API_KEY
from core.logging_config import get_logger

logger = get_logger(__name__)


class Mem0Service:
    """
    Memory backend using Mem0. Implements the same interface as VectorMemoryService
    so chat_service.py can swap between them transparently.
    """

    def __init__(self):
        from mem0 import Memory
        config = {}
        if MEM0_API_KEY:
            config["api_key"] = MEM0_API_KEY
        self._mem = Memory.from_config(config) if config else Memory()
        logger.info("[mem0] Mem0 memory backend initialized")

    async def store(self, db, user_id: str, content: str) -> None:
        """Store a memory for a user."""
        try:
            await asyncio.to_thread(self._mem.add, content, user_id=user_id)
            logger.info("[mem0] Stored memory for user %s: %r", user_id, content[:80])
        except Exception as exc:
            logger.warning("[mem0] Store failed for user %s: %s", user_id, exc)

    async def retrieve(self, db, user_id: str, query: str) -> list[str]:
        """Retrieve relevant memories for a user given a query."""
        try:
            results = await asyncio.to_thread(self._mem.search, query, user_id=user_id, limit=5)
            memories = []
            for r in results:
                if isinstance(r, dict):
                    memories.append(r.get("memory", r.get("text", str(r))))
                else:
                    memories.append(str(r))
            return memories
        except Exception as exc:
            logger.warning("[mem0] Retrieve failed for user %s: %s", user_id, exc)
            return []

    async def get_all(self, user_id: str) -> list[str]:
        """Get all memories for a user."""
        try:
            results = await asyncio.to_thread(self._mem.get_all, user_id=user_id)
            return [r.get("memory", str(r)) if isinstance(r, dict) else str(r) for r in results]
        except Exception as exc:
            logger.warning("[mem0] get_all failed: %s", exc)
            return []
```

**Step 3: Commit**

```bash
git add services/mem0_service.py core/config.py
git commit -m "feat: add Mem0Service with same interface as VectorMemoryService"
```

---

### Task 12: !memory command and chat_service swap

**Files:**
- Modify: `cogs/runtime_commands.py`
- Modify: `bot.py`
- Modify: `services/chat_service.py`

**Step 1: Add Mem0Service to bot.py**

Add import:
```python
from services.mem0_service import Mem0Service
```

In `__init__`:
```python
self.mem0_service = None
```

In `setup_hook`:
```python
from core.config import MEM0_ENABLED
if MEM0_ENABLED:
    try:
        self.mem0_service = Mem0Service()
    except Exception as exc:
        logger.warning("[startup] Mem0 failed to initialize: %s", exc)
```

**Step 2: Add !memory commands to runtime_commands.py**

```python
@commands.group(name="memory", invoke_without_command=True, help="Memory backend management.")
async def memory_group(self, ctx: commands.Context):
    from database.behavior_rules_repository import get_bot_config
    mode = await get_bot_config("memory_mode", "local")
    mem0 = getattr(self.bot, "mem0_service", None)
    mem0_status = "enabled" if mem0 is not None else "not enabled (set MEM0_ENABLED=true)"
    await ctx.send(
        f"Memory mode: `{mode}` (local=sqlite-vec, mem0=Mem0)\n"
        f"Mem0: {mem0_status}\n"
        f"Use `!memory mode local` or `!memory mode mem0` to switch."
    )

@memory_group.command(name="mode", help="Switch memory backend (local or mem0).")
async def memory_mode(self, ctx: commands.Context, mode: str):
    from database.behavior_rules_repository import set_bot_config
    mode = mode.strip().lower()
    if mode not in ("local", "mem0"):
        await ctx.send("Valid modes: `local`, `mem0`")
        return
    if mode == "mem0" and getattr(self.bot, "mem0_service", None) is None:
        await ctx.send("Mem0 is not enabled. Set `MEM0_ENABLED=true` in .env and restart.")
        return
    await set_bot_config("memory_mode", mode)
    await ctx.send(f"Memory backend switched to `{mode}`.")

@memory_group.command(name="status", help="Show memory stats for both backends.")
async def memory_status(self, ctx: commands.Context):
    from database.behavior_rules_repository import get_bot_config
    from database.db_connection import get_db
    mode = await get_bot_config("memory_mode", "local")
    user_id = str(ctx.author.id)

    lines = [f"Active backend: `{mode}`"]

    # Local stats
    try:
        async with get_db() as db:
            vec_service = getattr(self.bot, "vector_memory_service", None)
            if vec_service:
                from database.vector_memory_db import get_all_vector_memories
                rows = await get_all_vector_memories(db, user_id=user_id)
                lines.append(f"Local (sqlite-vec): {len(rows)} memories stored for you")
    except Exception as exc:
        lines.append(f"Local: error — {exc}")

    # Mem0 stats
    mem0 = getattr(self.bot, "mem0_service", None)
    if mem0:
        try:
            all_mems = await mem0.get_all(user_id)
            lines.append(f"Mem0: {len(all_mems)} memories stored for you")
        except Exception as exc:
            lines.append(f"Mem0: error — {exc}")
    else:
        lines.append("Mem0: not enabled")

    await ctx.send("\n".join(lines))
```

**Step 3: Update chat_service.py to swap memory backend**

In `generate_dynamic_reply`, find where `vector_memory_service` is used for retrieve and store. Replace with:

```python
# Select memory backend based on config
from database.behavior_rules_repository import get_bot_config
memory_mode = await get_bot_config("memory_mode", "local")
_mem0 = services.get("mem0_service")
_vec = services.get("vector_memory_service")
active_memory_service = _mem0 if (memory_mode == "mem0" and _mem0 is not None) else _vec
```

Then replace all `vector_memory_service` references in retrieve/store calls with `active_memory_service`.

Also add `mem0_service` to the services dict passed in from `chat_commands.py`:
```python
services["mem0_service"] = getattr(self.bot, "mem0_service", None)
```

**Step 4: Commit**

```bash
git add cogs/runtime_commands.py bot.py services/chat_service.py
git commit -m "feat: add !memory mode/status commands and swappable Mem0/local memory backend"
```

---

## Final Step: Update README

**Files:**
- Modify: `README.md`

Add new commands to the commands table and new feature bullets to Features section. Update `.env` variables table with all new config keys.

```bash
git add README.md
git commit -m "docs: update README with vision, thinking, coding, audio, and memory upgrade commands"
```

---

## Model Pull Reference

Before testing, pull models via Ollama:
```powershell
ollama pull moondream          # vision fast
ollama pull llava:34b          # vision best (needs ~20GB VRAM)
ollama pull deepseek-r1:7b     # think fast
ollama pull deepseek-r1:32b    # think best (~18GB VRAM)
ollama pull qwen2.5-coder:7b   # code ask fast
ollama pull devstral           # code ask best
```

Fish Speech and Parakeet require separate installs (see Task 8 and 9 prerequisites). Both are disabled by default (`FISH_SPEECH_ENABLED=false`, `PARAKEET_ENABLED=false`) — enable in `.env` after installing.
