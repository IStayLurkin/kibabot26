# Kiba Bot Changelog

All notable changes to this project are documented here.
Format: `[date] type: description` — grouped by release session.

---

## [2026-03-23] — Response Quality & Image Search Sprint

### New Features
- **Verified image search** — bot searches Giphy then a local folder when the user explicitly asks to see something ("show me cat memes", "let me see a funny cat gif", etc.). Every Giphy URL is scanned with VirusTotal before being sent. Unsafe results are silently skipped; if all fail, bot says so. Images are sent as Discord file attachments (8 MB cap enforced).
- **`!update` command** (owner only) — runs `git pull` and restarts the bot via `os.execv()`. Push a fix, type `!update` in Discord, done. No terminal required.

### Response Quality Fixes
- **Hallucinated URLs stripped** — any sentence containing a URL is dropped from LLM output before it reaches Discord. Prevents the model from sending fake `example.com` image links.
- **Farewell phrases stripped** — extended filler regex to catch `later alligator`, `see ya`, `catch you later`, `ttyl`, and similar goodbye phrases mid-conversation.
- **`feel free to keep chatting` caught** — extended filler closer regex to cover `keep chatting` / `chat` variants.
- **System prompt hardened** — LLM explicitly forbidden from: generating/guessing URLs, describing or captioning images, offering to show images unprompted.

### New Services
- `services/virustotal_service.py` — async VirusTotal v3 URL scanning (`is_safe(url)`). Submits URL, polls for completion, returns `True` only on 0 malicious + 0 suspicious detections. Fail-safe: returns `False` on any error or missing API key.
- `services/image_search_service.py` — `search_giphy(topic)`, `search_local(topic)`, `find_verified_image(topic)`. Local file matching handles both space and underscore filename variants.

### Detection
- `services/chat_router.py` — added `extract_image_request(text)`. Keyword + verb based detection handles natural phrasing: "show me", "send me", "got any", "let me see", "let me another", "can you show", etc.

### Config
- `core/config.py` — added `VIRUSTOTAL_API_KEY`, `GIPHY_API_KEY`, `LOCAL_IMAGE_DIR`.

---

## [2026-03-17] — Performance & Feature Sprint

### Performance
- **Shared SQLite connection with WAL mode** — replaced per-call `aiosqlite.connect()` with a single shared connection. WAL mode enables concurrent reads without blocking writes. Eliminated 10–50ms overhead per DB query and reduced SQLite lock contention under load.
- **Dedicated thread pool executors** — created `core/executors.py` with `HEAVY_EXECUTOR` (max 2 workers) for image/music generation and `LIGHT_EXECUTOR` (max 4 workers) for short ops. Prevents long-running model jobs from starving chat response threads.
- **Async hardware detection** — moved nvidia-smi subprocess calls in `vram_guard.py` off the event loop using `asyncio.to_thread()`. Prevents event loop blocking that was inflating Discord websocket ping.
- **Async startup** — moved `HardwareService._detect_status()` to `asyncio.to_thread()`. Eliminated synchronous torch CUDA init and nvidia-smi blocking the event loop during bot startup.
- **No duplicate hardware probes** — eliminated duplicate Ollama probe that was being called in both `refresh_hardware_status` and `sync_models("llm")` on startup.

### VRAM / RAM Fixes
- **Whisper auto-unload** — `VoiceService` now unloads the Whisper STT model after 5 minutes of inactivity (mirrors the existing image service pattern). Frees ~150MB VRAM when STT is idle.
- **Music pipeline auto-clear** — `MusicService` calls `clear_vram()` after every generation. StableAudio pipeline (~14GB) was previously staying in VRAM until bot restart.
- **SDXL VAE cast moved to load time** — was casting VAE to float32 on every single generation, temporarily doubling VAE VRAM usage (~2GB spike per run). Now done once at pipeline load.
- **Fixed hardcoded `[-10:]` slice** — `llm_service.py` was ignoring `CHAT_RECENT_MESSAGE_LIMIT = 8` with a hardcoded override. Context window now respects the constant.
- **Summary length cap** — conversation summaries are now capped at 1500 characters before storage. Prevents multi-KB summaries from bloating every Ollama request.
- **VRAM guard threshold lowered** — idle VRAM guard now triggers at 16384MB (16GB) instead of 21504MB (21.5GB). Gives headroom for model swaps before running out of room.
- **Summary trigger lowered** — `CHAT_SUMMARY_MIN_MESSAGES` reduced from 20 to 10. Summaries now compress history before messages start getting dropped from the context window.
- **Ollama eject uses config model name** — `MusicService._unload_ollama()` was hardcoded to eject `"qwen3-coder:7b"`. Now reads from `OLLAMA_MODEL` config so the correct model is always ejected before music generation.
- **asyncio.Lock in ImageService** — replaced `threading.Lock` with `asyncio.Lock`. Threading lock was being used in an async context and didn't coordinate correctly across async boundaries.
- **Voice STT on HEAVY_EXECUTOR** — `VoiceService.speech_to_text()` now uses `HEAVY_EXECUTOR` instead of the default unbounded thread pool, preventing duplicate Whisper instances under concurrent calls.

### New Features
- **Ollama/HF circuit breaker** — `services/circuit_breaker.py` tracks failures per provider. After 3 failures, skips that provider for 2 minutes before retrying. Prevents every chat message from hammering a dead provider.
- **Streaming-style reply delivery** — replies longer than 200 characters are sent as a placeholder message then progressively edited in 250-character chunks (50ms delay between edits). Makes the bot feel more responsive without waiting for the full generation.
- **Image prompt enhancement** — `!draw` and `!fast` commands now send the prompt through Ollama for enrichment before passing to FLUX/SDXL. Falls back to the original prompt on any failure.
- **`!forget` command** — any user can wipe their own chat history, memory, summary, and state for the current channel.
- **`!purge` command** (owner only) — wipes all chat history for every user in a channel.
- **`!models` command** — shows which Ollama models are currently loaded in VRAM via `/api/ps`.
- **`!allow` / `!deny` commands** (owner only) — add or remove channels from Kiba's allowed chat list at runtime. No restart required. Channel allowlist is now DB-backed and persists across restarts.
- **VRAM guard DM notification** — when the idle VRAM guard triggers a purge, it DMs the bot owner with before/after VRAM numbers.
- **Persistent cooldowns** — user chat cooldowns are now stored in the database and survive bot restarts.
- **Active session count on `!status`** — the dashboard now shows how many unique sessions have been active in the last 24 hours.

### Fixes
- **Fake second typing indicator** — `maybe_update_summary` was running inside the `async with destination.typing()` block, causing Discord to show a second typing indicator after every reply. Moved to `asyncio.create_task()` after the reply is sent.
- **Removed `asyncio` from `threading.Lock` context** — image generation lock corrected.
- **Removed unreachable debug print** in `video_service.py` after a `return` statement.
- **ComfyUI poll constants** — replaced magic numbers (`range(240)`, `time.sleep(1.0)`) with named constants `COMFYUI_POLL_MAX_ATTEMPTS` and `COMFYUI_POLL_INTERVAL_SECONDS`.

---

## [2026-03-14 to 2026-03-16] — Windows Migration & Audit Sprint

### Platform
- **Migrated from WSL2 to native Windows** — all paths updated from `/mnt/g/` to `G:/`. Bot now runs natively on Windows 11 with Python 3.12.9 at `G:\code\python\python.exe`.
- **Venv rebuilt** — old venv was pointing to a deleted Python installation at `C:\Users\btayl\AppData\Local\Programs\Python\Python312\`. Deleted and recreated from `G:\code\python\python.exe`.
- **`start_bot.ps1`** — replaced `start_bot.sh` with a PowerShell script that checks Ollama is running before launching.
- **Fixed OLLAMA_MODELS path** — was pointing to WSL path. Updated to `G:/ollamamodels`.
- **Fixed YuE repo path** — updated to `G:/code/python/learn_python/bot/YuE/inference`.

### Authentication
- **Fixed `discord.errors.LoginFailure`** — stale `DISCORD_BOT_TOKEN` was set as a Windows User environment variable, overriding `.env`. Fixed by adding `override=True` to `load_dotenv()`. Stale registry variable removed.

### OpenAI Removal
- **Stripped all OpenAI API usage** — removed all `OPENAI_*` config variables, imports, and client paths. Bot is now 100% local.
- **Provider chain: `ollama → hf` only** — `"openai"` removed from `ENABLED_MODEL_PROVIDERS`. Calling the OpenAI provider now raises `RuntimeError` immediately.
- **LLM fallback model fix** — when falling back to HuggingFace, was incorrectly passing the Ollama active model name (`kiba:latest`) to the HF API. Fixed `_get_model_for_provider` to use per-provider config defaults for non-active providers.
- **Removed from `model_storage_service.py`** — OpenAI removed from provider availability checks.

### Startup Performance
- **Ollama pre-warm** — added `_prewarm_ollama()` as `asyncio.create_task` on first `on_ready`. Sends a dummy request to load the model into VRAM before the first real user message. Eliminates 13+ second cold-start latency on first chat.
- **Removed 3-second countdown** from `main()` — was purely decorative, wasted startup time.

### Bug Fixes
- **`core/logging_config.py` indentation bug** — removed OpenAI logger line caused `discord.client` logger line to lose its indentation, breaking the module. Fixed.
- **`test_datetime_question_bypasses_llm` failure** — mock LLM was missing `timezone_name` attribute, causing `ZoneInfo` crash. Fixed by adding it to the test fixture.
- **`cogs/chat_commands.py`** — fixed `send_chunked` → `send_long_message` in `dossier` command. Added missing `_get_vram_usage()` method. Removed duplicate imports.
- **`services/agent_dispatcher.py`** — fixed eager construction of fallback services. Was always instantiating `ImageService`/`LLMService`/`MusicService` even when the bot already had them.
- **`services/voice_service.py`** — fixed unreachable docstring after executable code.
- **`kiba.modelfile`** — fixed typo `"ore Identity:"` → `"Core Identity:"`.

### Dependencies
- **Added `PyNaCl>=1.5.0,<2`** and **`davey>=0.1.4,<1`** to `requirements.txt` — required by discord.py 2.7.x for voice and DAVE E2EE protocol. Fixes `davey` warning on startup.
- **PyTorch upgraded** — `torch 2.6.0+cu124` → `2.7.1+cu128` to patch two CVEs.

### Docs & Housekeeping
- **`CODEBASE.md`** — added full file map, message flow diagram, and provider chain reference for future sessions.
- **`scripts/`** — moved `wipe_history.py` and `db_patch.py` from root to `scripts/`.
- **`.gitignore`** — removed dangerous `G*` glob. Added `*.log`, `.claude/settings.local.json`, `test_token.py`, `debug_gate.py`.
- **`debug_gate.py`** — guarded `uvloop` import to only run on non-Windows platforms.

---

## [2026-03-13] — Kiba Bot 2026 Foundation

### Core Architecture
- **Full local AI stack** — Ollama (Qwen3-Coder / kiba model) as primary LLM, HuggingFace as fallback. No cloud API dependencies.
- **RTX 3090 Ti hardware integration** — VRAM monitoring, nvidia-smi queries, CUDA-aware service initialization.
- **Multi-modal services** — image generation (FLUX.2 4-bit, SDXL FP16), music (YuE, StableAudio), voice (Piper TTS, Whisper STT), OSINT (WHOIS, DNS, SSL), code execution (sandboxed subprocess).
- **Agentic routing** — LangGraph-based `AgentDispatcher` routes intents to the correct service. `ToolRouter` handles keyword-based tool classification.
- **Conversation memory** — per-user/channel chat history, rolling summaries, extracted user memory facts, persistent conversation state.
- **`VRAMGuard` background task** — 5-minute loop monitors idle VRAM. Respects `bot.generating_count` to avoid clearing during active generation.
- **Performance tracking** — `PerformanceTracker` collects latency histograms for commands and service calls. Exposed via `!status` dashboard.

### Commands (initial)
- `!chat` / `!ask` / `!talk` — natural language chat via Ollama
- `!draw` — FLUX.2 image generation
- `!fast` / `!quick` — SDXL image generation
- `!status` / `!kiba` / `!kb` — GPU dashboard
- `!hardware` — real-time nvidia-smi stats
- `!boost` — manual VRAM cache clear
- `!dossier` / `!intel` / `!research` — OSINT research loop
- `!studio` — configure music generation settings (bpm, voice, mode)
- `!ping` / `!latency` — websocket latency
- `!about` — bot info
- `!vram` (owner) — VRAM status check

### Database
- SQLite via `aiosqlite` with tables for: expenses, budgets, chat sessions, chat messages, user memory, conversation summaries, conversation state, model registry, behavior rules, code execution logs.

---

## Earlier History

| Commit | Summary |
|--------|---------|
| `3db6714` | Added SDXL fast engine and fixed LLMService 3090 Ti initialization |
| `c12e94c` | Code quality pass: logging, routing, repo hygiene |
| `da132f9` | Real provider/runtime plumbing replacing stub registry entries |
| `62d03b3` | Database schema expanded |
| `f399d15` | Added Kiba agent system, media commands, services, config |
| `2dbcf5e` | Initial commit: modular Discord expense bot with AI chat |
