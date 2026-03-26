# Kiba Bot

Kiba is a high-performance, agentic Discord bot designed for **fully local execution**. Built on the **2026 local AI stack**, it leverages the **RTX 3090 Ti** to run unrestricted Large Language Models and high-fidelity image generation simultaneously.

## Features

- **Unrestricted Local Intelligence:** Powered by a custom Ollama `kiba` model (Dolphin 3.0 base — `dolphin3:8b-llama3.1-q8_0`). No content filtering.
- **Multi-Modal Generation:** FLUX.2 (4-bit) and SDXL (FP16) image generation; StableAudio instrumental; YuE full vocal tracks; CogVideoX-2b/5b, AnimateDiff, and Wan2.1 video generation.
- **Text-to-Speech / Speech-to-Text:** Tiered audio — `fast` uses Piper TTS + Faster-Whisper STT; `best` uses Fish Speech V1.5 (zero-shot voice cloning) + NVIDIA Parakeet V3 (sub-50ms STT).
- **Vision (Multimodal):** Attach an image to any message — bot auto-analyzes it with a vision model. `!vision fast` uses Moondream (2B, instant); `!vision best` uses LLaVA-34B (SOTA detail).
- **Thinking Models:** `!think fast` uses DeepSeek-R1:7B for quick chain-of-thought; `!think best` uses DeepSeek-R1:32B for deep reasoning.
- **Tiered Coding Assistant:** `!code ask fast` (Qwen2.5-Coder:7B) or `!code ask best` (Devstral-24B) for coding questions separate from the sandbox.
- **Persistent Memory:** Long-term and short-term memory stored locally — structured KV facts plus semantic/episodic vector memory (sqlite-vec + nomic-embed-text). Swappable Mem0 backend for comparison via `!memory mode mem0/local`.
- **Web Search (RAG):** Automatically searches the web via a local SearXNG instance when a message needs live information. Fast keyword pre-filter skips the classifier for casual chat; results injected as a grounded context block.
- **Agentic Routing:** LangGraph-based agent dispatcher routes natural language to image, music, video, code, or OSINT tools automatically.
- **Hardware Optimized:** Specifically tuned for **24GB VRAM** environments and **CUDA 12.8**. Automatic VRAM management — models unload after idle, VRAMGuard fires at 16GB used.
- **Verified Image Search:** Say "show me cat memes" — bot searches Giphy and your local image folder, scans every result with VirusTotal, and posts the first clean one.
- **Clean Response Pipeline:** Strips hallucinated URLs, fake image descriptions, filler openers/closers, farewell phrases, and emojis from all LLM output.
- **Sandboxed Code Execution:** `!code run` executes Python in an isolated workspace with dangerous-pattern detection.
- **Hot Update:** `!update` pulls latest code from GitHub and restarts the bot automatically — no terminal needed.
- **Live Startup Bar:** Terminal progress bar fills 0→100% during boot, logs print after fully ready.

## Tech Stack

- **Backend:** Python 3.12.9
- **LLM Engine:** Ollama (custom `kiba` model, Dolphin 3.0 base — `dolphin3:8b-llama3.1-q8_0`)
- **Image:** FLUX.2-dev (4-bit BnB), Stable Diffusion XL (FP16) via `diffusers`
- **Music:** StableAudio (instrumental), YuE via subprocess (full vocal tracks)
- **Video:** CogVideoX-2b/5b, AnimateDiff, Wan2.1
- **Voice:** Piper TTS (fast) / Fish Speech V1.5 (best), Faster-Whisper STT (fast) / NVIDIA Parakeet V3 (best)
- **Vision:** Moondream 2B (fast) / LLaVA-34B (best) via Ollama multimodal API
- **Thinking:** DeepSeek-R1:7B (fast) / DeepSeek-R1:32B (best) with chain-of-thought stripping
- **Coding:** Qwen2.5-Coder:7B (fast) / Devstral-24B (best) for `!code ask`
- **Agentic:** LangGraph `StateGraph` with async nodes
- **Frameworks:** discord.py 2.x, PyTorch (CUDA 12.8), Hugging Face Transformers
- **Memory:** sqlite-vec + nomic-embed-text (vector RAG), aiosqlite (structured KV + history)
- **Search:** SearXNG (self-hosted), Giphy API, VirusTotal API
- **Storage:** D: Drive (`D:\ai storage`) for all models and generated media — no G: or C: drive bloat

## Installation

1. **Clone the repo:**

   ```bash
   git clone https://github.com/IStayLurkin/kibabot26.git
   cd kibabot26
   ```

2. **Environment setup (PowerShell):**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   .\.venv\Scripts\pip install -r requirements.txt
   ```

   > Use `.venv\Scripts\python.exe` and `.venv\Scripts\pip.exe` directly to ensure the venv interpreter is used, not any system Python.

3. **Configure Ollama:**

   Ensure Ollama is installed and mapped to your high-capacity drive.

   Run the included Modelfile to bake the unrestricted personality:

   ```powershell
   ollama create kiba -f kiba.modelfile
   ```

4. **Configure environment variables:**

   Copy `.env.example` to `.env` and fill in your values:

   ```powershell
   Copy-Item .env.example .env
   ```

   Key variables:
   | Variable | Description |
   |---|---|
   | `DISCORD_BOT_TOKEN` | Your bot token |
   | `OLLAMA_MODEL` | Ollama model name (default: `dolphin-llama3:latest`) |
   | `GIPHY_API_KEY` | Giphy API key for image search |
   | `VIRUSTOTAL_API_KEY` | VirusTotal API key for URL scanning |
   | `LOCAL_IMAGE_DIR` | Optional local folder of trusted images/GIFs |
   | `SEARXNG_ENABLED` | `true`/`false` — enable web search (default: `false`) |
   | `SEARXNG_BASE_URL` | SearXNG instance URL (default: `http://localhost:8888`) |
   | `SEARXNG_MAX_RESULTS` | Max results per search query (default: `5`) |
   | `OLLAMA_NUM_CTX` | Ollama context window in tokens (default: `8192`) |
   | `THINKING_FAST_MODEL` | Fast thinking model (default: `deepseek-r1:7b`) |
   | `THINKING_BEST_MODEL` | Best thinking model (default: `deepseek-r1:32b`) |
   | `CODING_FAST_MODEL` | Fast coding model (default: `qwen2.5-coder:7b`) |
   | `CODING_BEST_MODEL` | Best coding model (default: `devstral:24b`) |
   | `VISION_FAST_MODEL` | Fast vision model (default: `moondream`) |
   | `VISION_BEST_MODEL` | Best vision model (default: `llava:34b`) |
   | `FISH_SPEECH_ENABLED` | Enable Fish Speech TTS (default: `false`) |
   | `FISH_SPEECH_BASE_URL` | Fish Speech server URL (default: `http://localhost:8080`) |
   | `PARAKEET_ENABLED` | Enable Parakeet STT (default: `false`) |
   | `MEM0_ENABLED` | Enable Mem0 memory backend (default: `false`) |
   | `MEM0_API_KEY` | Mem0 API key (leave empty for local mode) |

5. **Set up SearXNG (web search RAG):**

   Docker is required. The config is already included in `searxng_config/settings.yml`.

   ```powershell
   docker compose up -d searxng
   ```

   This is handled automatically by `start_bot.ps1` on every launch — no manual step needed after the first run.

   > Optionally change the `secret_key` in `searxng_config/settings.yml` from the default value.

6. **Run the bot:**

   ```powershell
   .\start_bot.ps1
   ```

   This automatically: starts Docker Desktop if needed → starts SearXNG on port 8888 → launches the bot.

   > **Note:** Ollama must already be running before executing `start_bot.ps1`. The bot does not auto-launch Ollama.

## Commands

Use `!help` or `!commands` in Discord to see the full live list at any time.

### Chat & Status

| Command | Access | Description |
|---|---|---|
| *(just talk)* | Everyone | Natural chat — routes automatically to LLM, image, music, video, or code |
| `!status` / `!kiba` / `!kb` | Everyone | GPU dashboard — VRAM bar, active engine, session count |
| `!hardware` | Everyone | Live nvidia-smi VRAM and CUDA stats |
| `!boost` | Everyone | Manual VRAM cache clear |
| `!models` | Everyone | Lists Ollama models currently loaded in VRAM |
| `!forget` | Everyone | Wipes your chat history, summary, and KV memory in this channel |
| `!forgetall @user` | Owner only | Full memory wipe — chat, summary, KV facts, vector/RAG memory |
| `!purge` | Owner only | Wipes all history for every user in the current channel |
| `!allow [channel]` | Owner only | Adds a channel to Kiba's allowed chat list |
| `!deny [channel]` | Owner only | Removes a channel from the allowed list |
| `!dossier <target>` | Everyone | 60s OSINT research loop on a person or topic |

### Image Generation

| Command | Access | Description |
|---|---|---|
| `!draw <prompt>` / `!image` | Everyone | FLUX.2 (4-bit) image — prompt auto-enhanced via Ollama |
| `!fast <prompt>` | Everyone | SDXL (FP16) image — faster, stylized |

### Vision

| Command | Access | Description |
|---|---|---|
| *(attach image to any message)* | Everyone | Auto-analyzed by fast vision model, context injected into reply |
| `!vision fast [prompt]` | Everyone | Analyze image with Moondream (fast, 2B) |
| `!vision best [prompt]` | Everyone | Analyze image with LLaVA-34B (best quality) |

### Thinking / Reasoning

| Command | Access | Description |
|---|---|---|
| `!think fast <prompt>` | Everyone | DeepSeek-R1:7B — quick chain-of-thought |
| `!think best <prompt>` | Everyone | DeepSeek-R1:32B — deep reasoning |

### Audio Generation

| Command | Access | Description |
|---|---|---|
| `!tts fast <text>` / `!say` | Everyone | Text-to-speech via Piper (fast, local) |
| `!tts best <text>` | Everyone | Text-to-speech via Fish Speech V1.5 (expressive, requires `FISH_SPEECH_ENABLED=true`) |
| `!stt fast` | Everyone | Switch voice transcription to Faster-Whisper |
| `!stt best` | Everyone | Switch voice transcription to Parakeet V3 (requires `PARAKEET_ENABLED=true`) |
| `!melody <prompt>` / `!music` / `!tune` | Everyone | Instrumental audio via StableAudio |
| `!song <vibe>. <lyrics>` / `!sing` | Everyone | Full vocal track via YuE (~6 min on 3090 Ti) |
| `!studio bpm/voice/mode <val>` | Owner only | Update music generation defaults |

### Video Generation

| Command | Access | Description |
|---|---|---|
| `!video <prompt>` / `!animate` | Everyone | Video (routes to default backend) |
| `!cogvideo2b <prompt>` | Everyone | CogVideoX-2b |
| `!cogvideo5b <prompt>` | Everyone | CogVideoX-5b |
| `!animatediff <prompt>` | Everyone | AnimateDiff |
| `!wan <prompt>` | Everyone | Wan2.1 |

### Code

| Command | Access | Description |
|---|---|---|
| `!code ask fast <prompt>` | Everyone | Ask Qwen2.5-Coder:7B a coding question |
| `!code ask best <prompt>` | Everyone | Ask Devstral-24B a coding question |
| `!code create <file> <content>` | Allowed users | Create a file in the sandbox |
| `!code edit <file> <content>` | Allowed users | Overwrite a file |
| `!code read <file>` | Allowed users | Read a file |
| `!code run <file>` | Allowed users | Run a Python file (3/min rate limit) |
| `!code list` | Allowed users | List sandbox files |
| `!code delete <file>` | Allowed users | Delete a file |
| `!code output <run_id>` | Allowed users | Fetch output from a previous run |

### Model Runtime

| Command | Access | Description |
|---|---|---|
| `!model [list/set/sync/pull/reload/add]` | Everyone | LLM model management |
| `!imagemodel [list/set/sync/pull/reload/add]` | Everyone | Image model management |
| `!audiomodel [list/set/sync/pull/reload/add]` | Everyone | Audio model management |
| `!cuda` / `!gpu` | Everyone | CUDA and GPU status |

### Behavior Rules

| Command | Access | Description |
|---|---|---|
| `!rule` | Everyone | List current behavior rules |
| `!rule add <text>` | Everyone | Add a persistent rule |
| `!rule edit <id> <text>` | Everyone | Edit a rule |
| `!rule delete <id>` | Everyone | Delete a rule |
| `!rule clear` | Everyone | Clear all rules |

### Personalities

Each user can set their own personality independently — changing yours won't affect anyone else's conversation.

| Command | Access | Description |
|---|---|---|
| `!personality` | Everyone | Show your current personality + available options |
| `!personality list` | Everyone | List all personalities with descriptions |
| `!personality set <name>` | Everyone | Switch your personality (persists across restarts) |
| `!personality reset` | Everyone | Reset yours to server default |
| `!personality global <name>` | Owner only | Change the server-wide default for all users |

Available personalities: `kiba` (default), `analyst`, `roast`, `tutor`, `hype`, `asian`, `dark`, `racist`, `bmw`, `weeb`, `midlife`, `therapist`

### OSINT (Owner/Admin)

| Command | Access | Description |
|---|---|---|
| `!osint <query>` | Owner/Admin | General OSINT lookup |
| `!whois <domain>` | Owner/Admin | WHOIS lookup |
| `!domain <domain>` | Owner/Admin | DNS + SSL certificate info |
| `!agent on/off/status` | Owner/Admin | Toggle agentic mode per channel |

### Memory

| Command | Access | Description |
|---|---|---|
| `!memory` | Everyone | Show active memory backend and status |
| `!memory mode local` | Everyone | Use sqlite-vec memory backend |
| `!memory mode mem0` | Everyone | Use Mem0 backend (requires `MEM0_ENABLED=true`) |
| `!memory status` | Everyone | Show memory counts from both backends |

### Admin / Dev

| Command | Access | Description |
|---|---|---|
| `!update` | Owner only | Git pull + hot restart |
| `!reload <cog>` | Owner only | Hot-reload a single cog |
| `!reloadall` | Owner only | Reload all managed extensions |
| `!reloadchat` | Owner only | Hot-reload chat_commands cog |

### Expenses & Budgets

| Command | Access | Description |
|---|---|---|
| `!expensehelp` | Everyone | Expense command reference |
| `!list` / `!ls` | Everyone | List all expenses |
| `!export` | Everyone | Export expenses to JSON |
| `!import_expenses` / `!imp` | Everyone | Import from expenses_import.json |

## Hardware Requirements

- **Minimum:** 12 GB VRAM
- **Recommended:** RTX 3090 Ti (24 GB VRAM) for simultaneous multi-modal agency

## License

MIT — see [LICENSE](LICENSE).
