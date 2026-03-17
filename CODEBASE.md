# Kiba Bot — Codebase Map

Quick reference for Claude. Point Claude here instead of crawling the repo.

## Entry Point
| File | Purpose |
|------|---------|
| `bot.py` | Main entry point, `ExpenseBot(commands.Bot)`, `setup_hook`, `on_message`, cog loading |
| `run_bot.ps1` | Launch script (Windows) |
| `watch_bot.ps1` | Auto-restart watcher |
| `kiba.modelfile` | Ollama model definition — Dolphin 3.0 fp16 base, kiba personality |

## core/
| File | Purpose |
|------|---------|
| `config.py` | All env vars — `LLM_PROVIDER`, `OLLAMA_*`, `HF_*`, feature flags. Load order matters: sets `OLLAMA_MODELS` before other imports |
| `constants.py` | Static constants (prefix, etc.) |
| `feature_flags.py` | Runtime feature toggles |
| `logging_config.py` | Colored console logger, Windows ANSI setup |
| `model_loaders.py` | Stub — model loader helpers |
| `utils.py` | Shared utilities |

## cogs/
Discord command handlers. All loaded in `bot.py:setup_hook`.

| File | Purpose |
|------|---------|
| `chat_commands.py` | `handle_natural_chat` — main AI chat entry point, calls `LLMService` |
| `agent_commands.py` | Agentic task commands |
| `media_commands.py` | Image/video/audio generation commands |
| `code_commands.py` | Code execution commands |
| `runtime_commands.py` | `!model`, `!hardware`, runtime switching |
| `dev_commands.py` | Developer/debug commands |
| `budget_commands.py` | Budget tracking |
| `expense_commands.py` | Expense tracking |
| `error_handler.py` | Global error handler cog |

## services/
| File | Purpose |
|------|---------|
| `llm_service.py` | Core LLM — provider chain (ollama → hf), `generate_reply`, `generate_agent_reply`, image/video stubs |
| `chat_service.py` | Chat pipeline — memory, context, routing to `LLMService` |
| `chat_router.py` | Routes messages to correct service |
| `agent_service.py` | Agentic loop logic |
| `agent_dispatcher.py` | Dispatches agent tool calls |
| `model_runtime_service.py` | Runtime state — active provider/model, hardware status |
| `model_storage_service.py` | Model storage, ollama pull |
| `hardware_service.py` | CUDA/VRAM/Ollama availability checks |
| `image_service.py` | Image generation (SDXL diffusers / automatic1111 / comfyui) |
| `voice_service.py` | Faster-Whisper STT, lazy-loaded |
| `video_service.py` | Video generation stub |
| `music_service.py` | YuE music generation via subprocess |
| `memory_service.py` | Short/long-term memory read/write |
| `summary_service.py` | Conversation summarization |
| `codegen_service.py` | Code generation |
| `code_execution_service.py` | Sandboxed code execution |
| `osint_service.py` | OSINT wrapper |
| `behavior_rule_service.py` | Persistent behavior rules |
| `performance_service.py` | Perf tracking / latency metrics |
| `tool_router.py` | Routes tool-use requests |
| `time_service.py` | Datetime context for prompts |
| `command_help_service.py` | Dynamic help text |
| `song_session_service.py` | Music session state |
| `media_safety_service.py` | Media safety checks |
| `expense_*.py` | Expense CRUD helpers (4 files) |

## database/
| File | Purpose |
|------|---------|
| `database.py` | SQLite init, `init_db()` |
| `chat_memory.py` | Short-term conversation memory (per-user) |
| `model_registry.py` | Model registry — `upsert_model`, `list_models`, `get_runtime_settings` |
| `behavior_rules_repository.py` | Persistent behavior rules store |
| `budget_repository.py` | Budget data |
| `execution_repository.py` | Code execution history |

## tasks/
| File | Purpose |
|------|---------|
| `task_manager.py` | Starts/stops all background tasks |
| `vram_guard.py` | VRAM monitor loop — frees memory when VRAM high |
| `health_tasks.py` | Periodic ping/health logging |

## osint_bot/
Self-contained OSINT sub-bot.

| File | Purpose |
|------|---------|
| `bot.py` | OSINT bot entry point |
| `core/config.py` | OSINT-specific config (`OSINT_OLLAMA_*`, `OSINT_HF_*`) |
| `services/llm_service.py` | OSINT LLM — ollama + hf only, uses openai SDK as HTTP client |
| `services/osint_service.py` | OSINT lookup logic |
| `cogs/osint_commands.py` | OSINT Discord commands |

## tests/
| File | Purpose |
|------|---------|
| `tests/test_code_execution_service.py` | Code sandbox tests (2 tests, both passing) |
| `osint_bot/tests/` | OSINT unit tests (boundary, formatting, policy, service, startup, validators) |

## Key Config Values (.env)
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=kiba:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
HF_MODEL=dolphin-llama3
DISCORD_BOT_TOKEN=...
```

## Message Flow
```
Discord message
  → bot.py on_message
  → cogs/chat_commands.py handle_natural_chat
  → services/chat_service.py
  → services/llm_service.py _build_provider_chain → [ollama, hf]
  → Ollama at http://127.0.0.1:11434/v1
  → response → send_long_message → Discord
```

## Provider Chain
- Primary: **ollama** (`kiba:latest` on RTX 3090 Ti)
- Fallback: **hf** (HuggingFace router)
- OpenAI: **disabled** — raises `RuntimeError` if called

## Notable Files (not Python)
| File | Purpose |
|------|---------|
| `kiba.modelfile` | Ollama modelfile — edit to change personality/params |
| `requirements.txt` | Pinned deps — torch cu128, discord.py, PyNaCl, davey |
| `.env` | Secrets and config — never committed |
| `bot.db` | SQLite database — never committed |
