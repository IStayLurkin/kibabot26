# Kiba Bot

Kiba is a high-performance, agentic Discord bot designed for **fully local execution**. Built on the **2026 local AI stack**, it leverages the **RTX 3090 Ti** to run unrestricted Large Language Models and high-fidelity image generation simultaneously.

## Features

- **Unrestricted Local Intelligence:** Powered by a custom Ollama `kiba` model (Dolphin 3.0 base).
- **GPU-Accelerated Rendering:** Local image generation using Stable Diffusion XL (SDXL) via the `diffusers` library.
- **Persistent Memory:** Long-term and short-term memory stored locally — structured KV facts plus semantic/episodic vector memory (sqlite-vec + nomic-embed-text). Top-5 relevant past memories injected into every reply.
- **Web Search (RAG):** Automatically searches the web via a local SearXNG instance when a message needs live information. Fast keyword pre-filter skips the classifier for casual chat; results injected as a grounded context block.
- **Hardware Optimized:** Specifically tuned for **24GB VRAM** environments and **CUDA 12.8**.
- **Verified Image Search:** Say "show me cat memes" — bot searches Giphy and your local image folder, scans every result with VirusTotal, and posts the first clean one as a Discord attachment.
- **Clean Response Pipeline:** Strips hallucinated URLs, fake image descriptions, filler openers/closers, farewell phrases, and emojis from all LLM output.
- **Hot Update:** `!update` pulls latest code from GitHub and restarts the bot automatically — no terminal needed.
- **Ollama Auto-Launch:** Bot starts `ollama serve` automatically on startup if it isn't already running.

## Tech Stack

- **Backend:** Python 3.12
- **LLM Engine:** Ollama (custom `kiba` model, Dolphin 3.0 base — `dolphin3:8b-llama3.1-q8_0`)
- **Frameworks:** Discord.py, PyTorch (CUDA 12.8), Hugging Face Transformers
- **Storage:** G: Drive Optimized (No C: drive bloat)

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

### Chat & Memory

| Command | Access | Description |
|---|---|---|
| `!forget` | Everyone | Clears your chat history, conversation summary, and KV memory in this channel |
| `!forgetall @user` | Owner only | Full memory wipe for a user — chat history, summary, KV facts, and RAG/vector memory |
| `!purge` | Owner only | Wipes all chat history for every user in the current channel |

### Expenses

| Command | Access | Description |
|---|---|---|
| `!clear yes` | Everyone | Clears all expenses |
| `!export` | Everyone | Exports expenses to a file |
| `!import_expenses` | Everyone | Imports expenses from a file |

### Admin

| Command | Access | Description |
|---|---|---|
| `!update` | Owner only | Pulls latest code from GitHub and restarts the bot |
| `!allow [channel]` | Owner only | Adds a channel to Kiba's allowed chat list |
| `!models` | Everyone | Shows which AI models are currently loaded in VRAM |

## Hardware Requirements

- **Minimum:** 12 GB VRAM
- **Recommended:** RTX 3090 Ti (24 GB VRAM) for simultaneous multi-modal agency

## License

MIT — see [LICENSE](LICENSE).
