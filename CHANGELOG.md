# Kiba Bot Changelog

All notable changes to this project are documented here.
Format: `[date] type: description` — grouped by release session.

---

## [2026-03-25] — Fine-Tooth Comb Pass #2

### Bug Fixes
- **`classify_intent()` logic was inverted** — `agent_dispatcher.py` returned `"chat"` when `"real"` was in the prompt (e.g. "sing me a real song") and routed to media otherwise. The intent was the opposite: "real" means actually generate it. Fixed both the music and media branches.
- **`workflow.ainvoke()` had no timeout** — if LangGraph stalled, the agentic chat path would hang indefinitely. Added `asyncio.wait_for(..., timeout=120.0)` with a logged error return.
- **VRAM bar `ZeroDivisionError`** — `!status` dashboard computed `int((used_vram / total_vram) * bar_length)` without guarding `total_vram == 0`. Added `if total_vram > 0 else 0`.
- **`GALLERY_CHANNEL_ID` int() unguarded** — a malformed env var would crash the entire image command path. Wrapped in `try/except (ValueError, TypeError)`.
- **STT lazy-load race condition** — `WhisperModel.__init__` was called without a lock; two concurrent STT requests could initialize two model instances simultaneously. Added `asyncio.Lock()` around the check-and-init block.
- **`HEAVY_EXECUTOR` import inside method** — `from core.executors import HEAVY_EXECUTOR` was inside `speech_to_text()` instead of at module level. Moved to top-level import.
- **YuE venv python path never validated** — `subprocess.run()` was called with a hardcoded `.venv/Scripts/python.exe` path that was never checked for existence. Now validates `venv_python.exists()` and returns early with a clear log message.
- **Discord file size never checked** — all five media commands (`!image`, `!tts`, `!video`, `!melody`, `!song`) sent files directly without checking size. Discord silently rejects uploads >25MB. Added `_check_file_size()` helper and applied it at all send sites.

---

## [2026-03-25] — Static Trace Audit & Command Fixes

### Bug Fixes
- **OSINT commands exceeded Discord 2000 char limit** — `!whois` truncated at 3900 chars then wrapped in a code block, pushing the total well past Discord's limit and causing a guaranteed send failure. `!domain` concatenated DNS + SSL output with no length check at all. `!osint` sent raw results with no truncation. All three now cap content at 1900 chars before wrapping.
- **Music agent node ignored user's prompt** — `music_agent_node` in `agent_dispatcher.py` hardcoded `vibe="cinematic"`, `bpm=120`, `voice_style="studio"` for every request regardless of what the user asked. Now extracts BPM from the prompt if specified (e.g. "120 bpm"), splits on `.` to separate vibe from lyrics (matching `!song` convention), and falls back to the music service's live configured defaults instead of arbitrary constants.
- **`!code create` gave cryptic error on path separators** — filenames containing `/` or `\` would reach the service's internal path traversal guard and surface a bare `ValueError`. Added an explicit upfront check with a clear user-facing message.

---

## [2026-03-25] — Help System & Final Bug Pass

### Bug Fixes
- **`VideoCommands` missing from help section labels** — `!cogvideo2b`, `!cogvideo5b`, `!animatediff`, `!wan` were displayed under the raw cog class name `"VideoCommands"` instead of a human-readable header. Now shows **"Video Generation"**.
- **Group parent commands invisible in `!help`** — `build_command_overview` iterated into subcommands and skipped the parent, so `!model`, `!imagemodel`, `!audiomodel`, `!rule`, and `!code` were completely absent. Now each group is listed with its subcommands indented beneath it.
- **`music_service` inconsistent failure returns** — `generate_melody` and `generate_song_clip` returned `None` on `TimeoutError` but `""` on all other exceptions. Unified to `None` on all failure paths; return types updated to `Optional[str]`.
- **`chat_service` passed `None` into `file_paths`** — when melody generation failed, `None` was placed directly in the `ChatReply.file_paths` list. Added null guard before building the reply.
- **`image_service` crashed with `AttributeError` on missing optional import** — if `diffusers`/`transformers` were not installed, `Flux2Transformer2DModel` and friends were `None`; calling `.from_pretrained()` on `None` produced an unhelpful `AttributeError`. Both `_load_flux()` and `_load_sdxl()` now raise a clear `RuntimeError` up front.
- **Fire-and-forget tasks in `image_service` and `voice_service` swallowed exceptions** — `_inactivity_monitor` and `_stt_inactivity_monitor` tasks had no done-callbacks; any exception was silently lost. Added `_on_monitor_done` / `_on_stt_monitor_done` static callbacks that log via `logger.exception`.
- **Piper process not killed on general `Exception`** — the `proc.kill()` / `proc.wait()` cleanup block only ran on `TimeoutError`. If `create_subprocess_exec` or `communicate` raised anything else, the process was left running. Kill logic now also runs in the `except Exception` branch.
- **`import torch, gc` inside `_stt_inactivity_monitor`** — deferred imports inside a method body can mask `ImportError` at unexpected times. Moved to module-level `try/except ImportError` block; `torch` is `None`-guarded where used.
- **`discord.File` opened without verifying file exists** — `handle_image_request` checked `if path:` but not `if Path(path).exists()`. If the output file was deleted between generation and upload, `discord.File()` would raise. Changed to `if path and Path(path).exists():`.

### Changes
- **`"Image Controls"` section renamed to `"Media"`** — `MediaCommands` covers TTS, music, and video routing in addition to images; the old label was misleading.

---

## [2026-03-25] — Quality & Reliability Improvements

### New Features
- **`safe_task()` utility** — all fire-and-forget `asyncio.create_task()` calls now use `safe_task()`, which attaches a done-callback that logs any unhandled exceptions. Previously `maybe_update_summary` and `_prewarm_ollama` failures were completely invisible.
- **Startup service validation** — on `on_ready`, bot now background-validates Giphy API key, SearXNG reachability, and embedding service responsiveness. Logs clear warnings instead of silently failing on first use.

### Bug Fixes
- **animatediff numpy import inside thread** — `import numpy as np` was inside `_generate_sync` (a `ThreadPoolExecutor` method). Moved to top-level `try/except ImportError` block alongside other optional imports.
- **STT executor call had no timeout** — `loop.run_in_executor(HEAVY_EXECUTOR, transcribe)` in `voice_service` could hang indefinitely if Whisper stalled. Added `asyncio.wait_for(..., timeout=120.0)`.
- **Music generation had no timeout** — YuE and StableAudio `run_in_executor` calls in `music_service` had no timeout. Added `asyncio.wait_for(..., timeout=480.0)` with `TimeoutError` handling that returns `None` and logs clearly.

---

## [2026-03-25] — Async & Threading Bug Fix Sprint

### Bug Fixes

- **`search_service` wired into chat pipeline** — `search_service` was created in `bot.py` but never stored on the bot instance or passed to `ChatCommands._build_services`. Web search RAG was silently inactive on the natural chat path. Now stored as `bot.search_service` and included in the services dict.
- **`last_update_time` race condition in `ImageService`** — initialized inside `_generate_sync` (called from a thread) instead of `__init__`, causing potential `AttributeError` on first callback and race on concurrent calls. Moved to `__init__`.
- **`get_event_loop()` in thread executor** — `_generate_sync` called `asyncio.get_event_loop()` from inside a `ThreadPoolExecutor`, which is broken in Python 3.12. The running loop is now captured in the async `_run_gen` and passed as a parameter.
- **`router_node` not async in `AgentDispatcher`** — LangGraph `ainvoke()` expects async nodes; `router_node` was a plain `def`. Made `async def`.
- **`bot.loop` deprecated access** — `handle_image_request` (chat_commands) and `_handle_video` (video_commands) both used `self.bot.loop` inside sync callbacks. Replaced with `_loop = asyncio.get_running_loop()` captured before entering the thread.
- **`bot.loop.create_task` in `health_tasks`** — replaced with `asyncio.get_running_loop().create_task(...)`.
- **`get_event_loop()` in `main()`** — replaced with `get_running_loop()` inside the async function.
- **`proc.kill()` unguarded in Piper TTS timeout** — `proc` could be unbound if `create_subprocess_exec` raised before assignment. Added `proc = None` guard and `await proc.wait()` to properly reap the process.
- **`hardware_service` null crash in `!models`** — added early return if `hardware_service` is None.
- **aiohttp image download had no timeout** — added `ClientTimeout(total=10)` to prevent indefinite hang on dead CDN.
- **`!update` skipped `bot.close()`** — `os.execv` was called immediately, bypassing DB close, Ollama process teardown, and WebSocket cleanup. Now calls `await self.bot.close()` first.
- **STT temp file used predictable name** — `temp_{filename}` in cwd could collide under concurrent uploads. Now uses `tempfile.gettempdir()` + UUID.
- **`__import__('numpy')` inline in `cogvideo_service`** — replaced with a proper `import numpy as np` in the `try/except ImportError` block at the top of the file.

---

## [2026-03-23] — Memory Management & Summary Fix

### New Features
- **`!forgetall <user_id>` command (owner only)** — full memory wipe for a user: clears chat history, conversation summary, KV facts, and RAG/vector memory. Complements `!forget` which only clears session-scoped memory.

### Bug Fixes
- **Conversation summary no longer includes bot capability descriptions** — the summary LLM prompt now explicitly instructs the model to focus only on facts about the user, preventing lines like "I can chat, joke, answer questions" from leaking into user memory summaries.

### Documentation
- **Commands section added to README** — documents all chat, expense, and admin commands with access levels.

---

## [2026-03-23] — Capabilities Trigger & RAG Improvements

### Bug Fixes
- **Capabilities trigger now matches "abilities"** — `matches_natural_language_help` and the `chat_service` routing both now catch "abilities" / "what abilities do you have" / "what are your abilities" variants, in addition to existing "capabilities" / "what can you do" triggers. Typos still fall through to the LLM but correctly-spelled ability queries now always hit `build_capabilities_summary`.

### Improvements
- **RAG similarity threshold** — `VectorMemoryService.retrieve` now filters results below `0.3` cosine similarity before returning. Previously all stored memories were injected into the prompt regardless of relevance; now only genuinely related memories surface.
- **RAG logging** — `store` and `retrieve` now emit `INFO` log lines showing what was stored and how many memories were retrieved vs total stored, making it easy to verify the RAG pipeline is active.

---

## [2026-03-23] — Log Truncation Fix

### Bug Fixes
- **Extended chat log truncation from 300 → 500 chars** — user and bot messages were being cut mid-sentence in the log output. Bumped both `[chat] user` and `[chat] bot` log slices to 500 chars to cover typical reply lengths without flooding logs.

---

## [2026-03-23] — Chat Context Bug Fix

### Bug Fixes
- **Fixed duplicate user message in LLM context** — `handle_chat_turn` was storing the user message to the DB before calling `generate_dynamic_reply`. Inside `generate_dynamic_reply`, `get_recent_chat_messages` fetched it back as part of history, and then `_build_messages` appended the current user message again. The LLM received two consecutive `user` role entries with the same text, causing it to lose conversational context and respond as if starting a fresh conversation. Fixed by moving `add_chat_message("user", ...)` to after `generate_dynamic_reply` returns for the text chat path. Image search and media paths store the user message immediately (unchanged behavior).

---

## [2026-03-23] — Web Search, Semantic Memory & Infrastructure Sprint

### New Features
- **Web search (SearXNG RAG)** — bot automatically decides when a message needs a live web search. A fast regex pre-filter skips the classifier for casual chat (~zero overhead). When needed, an LLM classifier generates up to 3 targeted queries, fires them in parallel against a local SearXNG instance, and injects the results as a `[SEARCH RESULTS]` block into the system prompt before generating the reply.
- **Semantic/episodic memory (Layer 1 RAG)** — every conversation turn is now semantically indexed. On each message, the user's text is embedded via `nomic-embed-text` (Ollama), and the top-5 most relevant past memories are retrieved by cosine similarity and injected as a `[RELEVANT MEMORIES]` block. After each reply, a background task asks the LLM what's worth remembering and stores it as a vector embedding. Runs fully local — no external servers. Existing KV fact store unchanged.
- **Ollama auto-launch** — `bot.py` checks if `ollama serve` is running on startup and launches it automatically if not. Polls up to 30s for readiness. Terminates the managed process cleanly on bot shutdown.

### New Services
- `services/search_service.py` — `SearchService` with `search()` and `search_many()` (parallel asyncio.gather). Wraps SearXNG JSON API.
- `services/embedding_service.py` — `EmbeddingService` wrapping Ollama `/api/embeddings` via httpx. `embed()` and `embed_many()`.
- `services/vector_memory_service.py` — `VectorMemoryService` with `store()` (embed + write) and `retrieve()` (embed + cosine rank + top-K). Pure Python cosine similarity, no external vector DB.

### New DB
- `database/vector_memory_db.py` — `vector_memories` table (`user_id`, `content TEXT`, `embedding BLOB`, `created_at`). Helpers: `store_vector_memory`, `get_all_vector_memories`, `delete_vector_memories`. Embedding dimension guard (768-dim, nomic-embed-text).
- `database/db_connection.py` — loads `sqlite-vec` extension on connection open.

### LLMService Extensions
- `_classify_search_need(message)` — LLM classifier returning up to 3 search query strings.
- `_message_needs_search(message)` — module-level regex pre-filter; skips classifier entirely for casual chat.
- `extract_episodic_memory(user_message, bot_reply)` — LLM decides what's worth storing; returns `{should_store, content}`.
- `_build_messages` now accepts `search_results` and `relevant_memories`; injects both as blocks in the system prompt.
- `generate_agent_reply` now accepts and passes through `relevant_memories`.

### Config
- `core/config.py` — added `SEARXNG_ENABLED`, `SEARXNG_BASE_URL`, `SEARXNG_MAX_RESULTS`.
- `.env` — added SearXNG defaults (gitignored).

### Infrastructure
- `searxng_config/settings.yml` — Docker volume config for SearXNG: JSON format enabled, binds to `0.0.0.0:8080`, rate limiter off.

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
