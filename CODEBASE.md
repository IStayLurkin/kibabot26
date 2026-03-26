# Kiba Bot — Codebase Map

Quick reference for Claude. Point Claude here instead of crawling the repo.

---

## Entry Point
| File | Purpose |
|------|---------|
| `bot.py` | Main entry point, `ExpenseBot(commands.Bot)`, `setup_hook`, `on_message`, cog loading |
| `start_bot.ps1` | Launch script (Windows) — starts Docker Desktop if needed, starts SearXNG on port 8888, then launches bot. Ollama must already be running. |
| `watch_bot.ps1` | Auto-restart watcher |
| `kiba.modelfile` | Ollama model definition — Dolphin 3.0 q8_0 base, kiba personality. Never greets or volunteers date/time unprompted. |

---

## core/
| File | Purpose |
|------|---------|
| `config.py` | All env vars — `LLM_PROVIDER`, `OLLAMA_*`, `HF_*`, feature flags. `override=True` on `load_dotenv()` prevents stale Windows env vars from winning. Validates `DEFAULT_MODEL_PROVIDER` is in `ENABLED_MODEL_PROVIDERS` at import. Warns if both `CODE_ALLOWED_USER_IDS` and `CODE_ALLOWED_ROLE_IDS` are empty. |
| `constants.py` | Static constants — cooldowns, chat limits, version. `CHAT_RECENT_MESSAGE_LIMIT=8`, `CHAT_SUMMARY_MIN_MESSAGES=10`. |
| `executors.py` | Thread pool executors — `HEAVY_EXECUTOR` (max 2, for image/music/STT), `LIGHT_EXECUTOR` (max 4, for short ops). Import from here, never use `asyncio.to_thread` for heavy jobs. |
| `feature_flags.py` | Runtime feature toggles |
| `logging_config.py` | Colored console logger, Windows ANSI setup |
| `model_loaders.py` | Stub — model loader helpers |
| `utils.py` | Shared utilities |

---

## cogs/
Discord command handlers. All loaded in `bot.py:setup_hook`.

| File | Commands | Notes |
|------|----------|-------|
| `chat_commands.py` | `!status`/`!kiba`/`!kb`, `!hardware`, `!boost`, `!draw`, `!fast`, `!dossier`, `!studio`, `!forget`, `!forgetall`, `!purge`, `!models`, `!allow`, `!deny` | Main AI chat entry point. `handle_natural_chat` → `handle_chat_turn`. Streaming-style reply delivery for long responses. VRAM bar guards `total_vram == 0`. `GALLERY_CHANNEL_ID` int() wrapped in try/except. |
| `agent_commands.py` | `!agent on/off/status`, `!osint`, `!whois`, `!domain` | OSINT outputs truncated to 1900 chars before Discord code block wrapper. |
| `media_commands.py` | `!image`/`!img`, `!tts`/`!say`, `!video`/`!animate`, `!melody`/`!music`/`!tune`, `!song`/`!vocals`/`!sing` | All five send sites check file size ≤25MB before `discord.File()`. |
| `code_commands.py` | Code execution commands. `!code run` has a 3/60s per-user cooldown. `!code read` truncation uses `CODE_MAX_OUTPUT_CHARS` from config. | — |
| `runtime_commands.py` | `!model`, `!imagemodel`, `!audiomodel` groups + `!cuda`/`!gpu`, `!commands`, `!help`, `!rule` group, `!personality` group. No duplicate `switch` subcommands. | — |
| `dev_commands.py` | Developer/debug commands | — |
| `budget_commands.py` | Budget tracking | — |
| `expense_commands.py` | Expense tracking | — |
| `error_handler.py` | Global error handler cog. Handles: `CommandNotFound`, `MissingRequiredArgument`, `BadArgument`, `CommandOnCooldown` (with retry seconds), `NotOwner`, `CheckFailure`, `MissingPermissions`. | — |

### Key Commands Reference
| Command | Who | What |
|---------|-----|------|
| `!status` / `!kiba` / `!kb` | anyone | GPU dashboard — VRAM, active engine, session count |
| `!hardware` | anyone | Real-time nvidia-smi stats |
| `!boost` | anyone | Manual VRAM cache clear |
| `!draw <prompt>` | anyone | FLUX.2 image generation (prompt auto-enhanced via Ollama) |
| `!fast <prompt>` | anyone | SDXL image generation (prompt auto-enhanced via Ollama) |
| `!image <prompt>` / `!img` | anyone | Alias for `!draw` via MediaCommands cog |
| `!tts <text>` / `!say` | anyone | Text-to-speech via Piper |
| `!video <prompt>` / `!animate` | anyone | Video generation (CogVideoX default backend) |
| `!melody <prompt>` / `!music` / `!tune` | anyone | Instrumental audio via StableAudio |
| `!song <vibe>. <lyrics>` / `!vocals` / `!sing` | anyone | Full vocal track via YuE (~6 min) |
| `!cogvideo2b <prompt>` | anyone | CogVideoX-2b video generation |
| `!cogvideo5b <prompt>` | anyone | CogVideoX-5b video generation |
| `!animatediff <prompt>` | anyone | AnimateDiff video generation |
| `!wan <prompt>` | anyone | Wan2.1 video generation |
| `!models` | anyone | Lists Ollama models currently loaded in VRAM |
| `!forget` | anyone | Wipes own chat history + memory in current channel |
| `!forgetall @user` | owner | Full memory wipe — chat, summary, KV, vector/RAG |
| `!purge` | owner | Wipes all history for every user in current channel |
| `!allow [channel]` | owner | Add channel to allowed list (no restart needed) |
| `!deny [channel]` | owner | Remove channel from allowed list (no restart needed) |
| `!dossier <target>` | anyone | OSINT research loop (60s cooldown) |
| `!studio bpm/voice/mode <val>` | owner | Update music generation settings |
| `!osint <query>` | owner/admin | OSINT lookup |
| `!whois <domain>` | owner/admin | WHOIS lookup |
| `!domain <domain>` | owner/admin | DNS + SSL info |
| `!agent on/off/status` | owner/admin | Toggle agentic mode per channel |
| `!model [list/set/sync/...]` | anyone | LLM model management group |
| `!imagemodel [...]` | anyone | Image model management group |
| `!audiomodel [...]` | anyone | Audio model management group |
| `!rule [add/edit/delete/clear]` | anyone | Persistent bot behavior rules |
| `!code [create/edit/run/read/list/delete/output]` | allowed users | Sandboxed Python execution |
| `!cuda` / `!gpu` | anyone | CUDA and GPU status |
| `!help [command]` | anyone | Dynamic help — lists all commands or details for one |
| `!commands` / `!cmds` | anyone | Full command list |
| `!personality` | anyone | Show your current personality |
| `!personality set <name>` | anyone | Set your personal personality (per-user, persists) |
| `!personality reset` | anyone | Reset yours to server default |
| `!personality list` | anyone | List all personalities with descriptions |
| `!personality global <name>` | owner | Set the server-wide default personality |
| `!update` | owner | Git pull + restart |
| `!reload <cog>` | owner | Hot-reload a cog |
| `!reloadall` | owner | Reload all managed extensions |

---

## services/
| File | Purpose | Notes |
|------|---------|-------|
| `llm_service.py` | Core LLM — provider chain (ollama → hf), `generate_reply`, `generate_agent_reply`, `enhance_image_prompt` | Circuit breakers per provider. `_build_provider_chain()` filters tripped providers. `PERSONALITIES` dict holds 12 named system prompts. `generate_reply()` accepts `personality` param; `_build_messages()` resolves it: per-call arg → `active_personality` instance fallback. `_FILLER_OPENING` regex strips full sentence (not just opener word) to prevent dangling fragments. |
| `circuit_breaker.py` | Per-provider failure tracking — opens after 3 failures, recovers after 2min cooldown | Used inside `LLMService`. |
| `chat_service.py` | Chat pipeline — memory, context, routing to `LLMService` | Builds full context: system prompt + memories + summary + history. |
| `chat_router.py` | Routes messages to correct service | — |
| `agent_service.py` | Agentic loop logic | — |
| `agent_dispatcher.py` | Dispatches agent tool calls. Increments `bot.generating_count` during media generation. | `generating_count` is incremented before `try`, decremented in `finally` — both `media_node` and `music_agent_node`. Increment and `try` are adjacent with no `await` between them, so the decrement always runs on any exception. No leak. `workflow.ainvoke()` wrapped in `asyncio.wait_for(timeout=120s)` — stall returns a clean error message. `classify_intent()`: music/draw triggers route to generation unless `"real"` is absent AND creative override is absent — "real" means generate it for real. `music_agent_node` extracts BPM from prompt, splits vibe/lyrics on `.`, falls back to service config defaults. |
| `model_runtime_service.py` | Runtime state — active provider/model, hardware status | `initialize()` fetches hardware once, passes to `sync_models` to avoid duplicate probes. |
| `model_storage_service.py` | Model storage, ollama pull | — |
| `hardware_service.py` | CUDA/VRAM/Ollama availability checks. `get_vram_usage_mb()` via nvidia-smi. `get_ollama_running_models()` via `/api/ps`. | All blocking calls wrapped in `asyncio.to_thread`. |
| `image_service.py` | FLUX.2 (4-bit) and SDXL (FP16) generation. Auto-purges after 5min idle. | Uses `HEAVY_EXECUTOR`. `asyncio.Lock` acquired before dispatch. SDXL VAE cast to float32 at load time. `_load_flux()`/`_load_sdxl()` raise `RuntimeError` immediately if optional diffusers imports failed (prevents cryptic `AttributeError` on `None`). Inactivity monitor task has done-callback for exception visibility. |
| `voice_service.py` | Piper TTS + Faster-Whisper STT. Whisper lazy-loaded, auto-unloads after 5min idle. | STT uses `HEAVY_EXECUTOR`. `_stt_lock` (asyncio.Lock) prevents double-init of WhisperModel on concurrent calls. Piper process killed in both `TimeoutError` and general `Exception` handlers. STT inactivity monitor task has done-callback for exception visibility. |
| `video_service.py` | Video generation stub (disabled) | — |
| `music_service.py` | StableAudio (melody) + YuE subprocess (full songs). `clear_vram()` called after every generation. Ejects Ollama model before loading. | Uses `HEAVY_EXECUTOR`. Ollama eject reads the live model name from `model_runtime_service.get_active_llm_model()`, falls back to `OLLAMA_MODEL` config if runtime unavailable. Pass `runtime_service=` at construction. YuE validates `.venv/Scripts/python.exe` exists before subprocess. Both generation methods return `Optional[str]` — `None` on all failure paths. |
| `memory_service.py` | Short/long-term memory read/write. Memory values >20 words are skipped (logged at DEBUG). `blocked_memory_keys` is an exact-match set on the LLM-extracted `memory_key`: `{"income", "salary", "budget", "budget_categories", "groceries", "rent", "fun_money", "extra"}`. Not substring matching — the full key must match. | — |
| `summary_service.py` | Conversation summarization. Summaries capped at 1500 chars before storage. Triggers after `CHAT_SUMMARY_MIN_MESSAGES=10`. Known: if the bot restarts mid-summarization, the partial summary write is silently dropped — the previous summary remains. Low severity; next message cycle will re-trigger. | — |
| `codegen_service.py` | Code generation | — |
| `code_execution_service.py` | Sandboxed code execution. `DANGEROUS_PATTERNS` lowercased at definition — includes `requests`, `pickle.loads/load`, `eval`, `exec`, `subprocess`, etc. Uses full path in subprocess command. | — |
| `osint_service.py` | OSINT wrapper — WHOIS, DNS, SSL | — |
| `behavior_rule_service.py` | Persistent behavior rules. `extract_rule_replacement` safely guards the `" to "` split — returns `("", "")` if delimiter not present. | — |
| `performance_service.py` | Perf tracking / latency metrics | `SERVICE_WARNING_THRESHOLD_MS=10000`, `SERVICE_ERROR_THRESHOLD_MS=20000`. Media generation prefixes (`animatediff`, `cogvideo`, `wan`, `image`, `video`, `voice`, `music`) suppressed from slow-op warnings. |
| `tool_router.py` | Routes tool-use requests. `detect_tool` correctly skips image detection for non-media requests. Clarifying question threshold is ≤2 words — applied to the **full lowercased message**, not extracted intent. "draw a cat" (3 words) does not trigger clarification. Code markers are language-specific (`python`, `def `, `function(`, etc.) — not generic words like `class`/`bug`. | — |
| `time_service.py` | Datetime context for prompts | — |
| `command_help_service.py` | Dynamic help text. `build_command_overview` lists Group parent commands (e.g. `!model ...`) with subcommands indented beneath. `SECTION_LABELS` includes `VideoCommands → "Video Generation"`. | — |
| `song_session_service.py` | Music session state | — |
| `media_safety_service.py` | Media safety checks | — |
| `expense_*.py` | Expense CRUD helpers (4 files) | — |

---

## database/
Shared connection via `db_connection.py`. WAL mode enabled. Do not open new `aiosqlite.connect()` calls — always use `get_db()`.

| File | Purpose |
|------|---------|
| `db_connection.py` | Shared `aiosqlite` connection. `get_db()` lazily opens with WAL + foreign_keys + synchronous=NORMAL. `close_db()` called on bot shutdown. |
| `database.py` | SQLite init, `init_db()` — expense and budget tables |
| `chat_memory.py` | Chat sessions, messages, user memory, summaries, conversation state, cooldowns, allowed channels. All functions use `get_db()`. |
| `model_registry.py` | Model registry — `upsert_model`, `list_models`, `get_runtime_settings` |
| `behavior_rules_repository.py` | Persistent behavior rules store + `bot_config` table (generic key-value). `get_bot_config(key, default)` / `set_bot_config(key, value)`. Active personality stored as `user_personality:{user_id}` and `active_personality` (global default). |
| `budget_repository.py` | Budget data |
| `execution_repository.py` | Code execution history |

### DB Tables Quick Reference
| Table | Stores |
|-------|--------|
| `chat_sessions` | user_id + channel_id pairs |
| `chat_messages` | Per-session message history |
| `user_memory` | Key-value facts per user |
| `chat_summaries` | Rolling conversation summaries (capped 1500 chars) |
| `chat_state` | goal, intent, response_mode, last_tool, pending_question. `pending_question` is actively written in `chat_service.py` when clarification is needed (line 411, 465) and cleared on reply (line 431, 523, 549). Also injected into LLM context (line 447). Not orphaned. |
| `user_cooldowns` | Last used timestamp per user (survives restarts) |
| `allowed_channels` | DB-backed channel allowlist |

---

## tasks/
| File | Purpose |
|------|---------|
| `task_manager.py` | Starts/stops all background tasks |
| `vram_guard.py` | 5-min loop — fires at 16GB idle VRAM. Respects `bot.generating_count`. Logs silently (owner DM removed). |
| `health_tasks.py` | Periodic ping/health logging |

---

## tests/
| File | Tests |
|------|-------|
| `test_code_execution_service.py` | Sandbox path traversal, file create/read |
| `test_provider_chain.py` | Provider chain order, OpenAI disabled, runtime model selection, fallback behavior |
| `test_chat_routing.py` | ChatReply generation, datetime bypass, missing service degradation, runtime query |
| `test_db_connection.py` | Shared connection, singleton, close/reopen |
| `test_executors.py` | HEAVY_EXECUTOR and LIGHT_EXECUTOR are correct types and distinct |
| `test_circuit_breaker.py` | Opens after threshold, resets on success, recovers after cooldown |
| `test_streaming.py` | Chunk boundary logic |
| `test_prompt_enhancement.py` | enhance_image_prompt happy path and error fallback |
| `test_forget_command.py` | delete_user_history, delete_channel_history |
| `test_models_command.py` | get_ollama_running_models success and error paths |
| `test_memory_service.py` | extract_memory_fact, should_attempt_memory_storage, blocked finance keys, 20-word cap, AI extraction, store_memory_if_found |
| `test_chat_service.py` | ChatReply, agentic flag on/off, datetime bypass, behavior rules forwarding, memory context injection, missing LLM fallback |

Run all: `.venv\Scripts\python.exe -m pytest tests/ -v`
Currently: **64 passing**

---

## Key Config Values (.env)
```
DISCORD_BOT_TOKEN=...
LLM_PROVIDER=ollama
OLLAMA_MODEL=kiba:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
HF_MODEL=dolphin-llama3
GALLERY_CHANNEL_ID=...
GPU_TOTAL_VRAM_MB=24576
```

---

## Message Flow
```
Discord message
  → bot.py on_message
  → process_commands()          ← prefix commands (!draw, !forget, etc.)
  → if not valid command:
      → cogs/chat_commands.py handle_natural_chat
          → handle_chat_turn
              → AgentDispatcher.classify_intent
              → if draw/sing: AgentDispatcher.run → ImageService / MusicService
              → else: chat_service.generate_dynamic_reply
                  → ToolRouter (code / osint / image / voice / music)
                  → LLMService._build_provider_chain → [ollama, hf]
                  → Ollama at http://127.0.0.1:11434/v1
              → streaming-style reply (chunked edits for >200 char responses)
              → asyncio.create_task(maybe_update_summary)  ← background, non-blocking
```

---

## Provider Chain
- Primary: **ollama** (`kiba:latest` on RTX 3090 Ti)
- Fallback: **hf** (HuggingFace router)
- OpenAI: **disabled** — raises `RuntimeError` if called
- Circuit breaker: skips provider for 2 min after 3 consecutive failures

## VRAM Management
- **Image service**: purges on engine swap, auto-unloads after 5min idle
- **Music service**: `clear_vram()` called after every generation
- **Voice service**: Whisper unloads after 5min idle
- **VRAMGuard**: fires at 16GB idle, DMs owner, respects `bot.generating_count`
- **Before music gen**: ejects active Ollama model via `/api/chat keep_alive=0`
- **Thread isolation**: heavy jobs (image/music/STT) use `HEAVY_EXECUTOR` (max 2 workers)

---

## Notable Files (not Python)
| File | Purpose |
|------|---------|
| `kiba.modelfile` | Ollama modelfile — edit to change personality/params. Rebuild with `ollama create kiba -f kiba.modelfile` |
| `requirements.txt` | Pinned deps — torch cu128, discord.py, PyNaCl, davey |
| `CHANGELOG.md` | Full project history organized by session |
| `.env` | Secrets and config — never committed |
| `bot.db` | SQLite database — never committed |
| `docs/plans/` | Implementation plans for major feature sprints |
| `docker-compose.yml` | SearXNG service definition — `kiba-searxng` container on port 8080, mounts `searxng_config/` |
| `searxng_config/settings.yml` | SearXNG config — port 8080, limiter disabled, JSON format enabled, `secret_key` should be changed for production |

## Startup Scripts
| File | Purpose |
|------|---------|
| `start_bot.ps1` | **Main launch script.** Starts Ollama if not running → starts SearXNG via Docker if not running → launches bot |
| `run_bot.ps1` | Alternate launch (no SearXNG check) |
| `watch_bot.ps1` | Auto-restart watcher |
