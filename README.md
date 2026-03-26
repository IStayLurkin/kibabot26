# Kiba Bot

Kiba is a high-performance, agentic Discord bot designed for **fully local execution**. Built on the **2026 local AI stack**, it leverages the **RTX 3090 Ti** to run unrestricted Large Language Models and high-fidelity image generation simultaneously.

## Features

- **Unrestricted Local Intelligence:** Powered by a custom Ollama `kiba` model (Dolphin 3.0 base — `dolphin3:8b-llama3.1-q8_0`). No content filtering.
- **Multi-Modal Generation:** FLUX.2 (4-bit) and SDXL (FP16) image generation; StableAudio instrumental; YuE full vocal tracks; CogVideoX-2b/5b, AnimateDiff, and Wan2.1 video generation.
- **Text-to-Speech / Speech-to-Text:** Piper TTS (local ONNX model) + Faster-Whisper STT. Upload a voice clip and Kiba transcribes and replies.
- **Persistent Memory:** Long-term and short-term memory stored locally — structured KV facts plus semantic/episodic vector memory (sqlite-vec + nomic-embed-text). Top-5 relevant past memories injected into every reply.
- **Web Search (RAG):** Automatically searches the web via a local SearXNG instance when a message needs live information. Fast keyword pre-filter skips the classifier for casual chat; results injected as a grounded context block.
- **Agentic Routing:** LangGraph-based agent dispatcher routes natural language to image, music, video, code, or OSINT tools automatically.
- **Hardware Optimized:** Specifically tuned for **24GB VRAM** environments and **CUDA 12.8**. Automatic VRAM management — models unload after idle, VRAMGuard fires at 16GB used.
- **Verified Image Search:** Say "show me cat memes" — bot searches Giphy and your local image folder, scans every result with VirusTotal, and posts the first clean one.
- **Clean Response Pipeline:** Strips hallucinated URLs, fake image descriptions, filler openers/closers, farewell phrases, and emojis from all LLM output.
- **Sandboxed Code Execution:** `!code run` executes Python in an isolated workspace with dangerous-pattern detection.
- **Hot Update:** `!update` pulls latest code from GitHub and restarts the bot automatically — no terminal needed.
- **Ollama Auto-Launch:** Bot starts `ollama serve` automatically on startup if it isn't already running.

## Tech Stack

- **Backend:** Python 3.12.9
- **LLM Engine:** Ollama (custom `kiba` model, Dolphin 3.0 base — `dolphin3:8b-llama3.1-q8_0`)
- **Image:** FLUX.2-dev (4-bit BnB), Stable Diffusion XL (FP16) via `diffusers`
- **Music:** StableAudio (instrumental), YuE via subprocess (full vocal tracks)
- **Video:** CogVideoX-2b/5b, AnimateDiff, Wan2.1
- **Voice:** Piper TTS (ONNX), Faster-Whisper STT (CUDA)
- **Agentic:** LangGraph `StateGraph` with async nodes
- **Frameworks:** discord.py 2.x, PyTorch (CUDA 12.8), Hugging Face Transformers
- **Memory:** sqlite-vec + nomic-embed-text (vector RAG), aiosqlite (structured KV + history)
- **Search:** SearXNG (self-hosted), Giphy API, VirusTotal API
- **Storage:** G: Drive optimized (no C: drive bloat)

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
   | `SEARXNG_BASE_URL` | SearXNG instance URL (default: `http://localhost:8080`) |
   | `SEARXNG_MAX_RESULTS` | Max results per search query (default: `5`) |

5. **Run the bot:**

   ```powershell
   .\run_bot.ps1
   ```

   This checks Ollama is running first, then launches the bot and tees output to `bot.log`.

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

### Audio Generation

| Command | Access | Description |
|---|---|---|
| `!tts <text>` / `!say` | Everyone | Text-to-speech via local Piper |
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

### Code Sandbox

| Command | Access | Description |
|---|---|---|
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

### OSINT (Owner/Admin)

| Command | Access | Description |
|---|---|---|
| `!osint <query>` | Owner/Admin | General OSINT lookup |
| `!whois <domain>` | Owner/Admin | WHOIS lookup |
| `!domain <domain>` | Owner/Admin | DNS + SSL certificate info |
| `!agent on/off/status` | Owner/Admin | Toggle agentic mode per channel |

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
