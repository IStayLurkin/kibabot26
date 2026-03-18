# Kiba Bot

Kiba is a high-performance, agentic Discord bot designed for **fully local execution**. Built on the **2026 local AI stack**, it leverages the **RTX 3090 Ti** to run unrestricted Large Language Models and high-fidelity image generation simultaneously.

## Features

- **Unrestricted Local Intelligence:** Powered by a custom Ollama `kiba` model (Dolphin 3.0 base).
- **GPU-Accelerated Rendering:** Local image generation using Stable Diffusion XL (SDXL) via the `diffusers` library.
- **Persistent Memory:** Long-term and short-term memory stored locally to recall user facts and conversation context.
- **Hardware Optimized:** Specifically tuned for **24GB VRAM** environments and **CUDA 12.8**.

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

   Copy `.env.example` to `.env` and fill in your Discord token and any other values.

   ```powershell
   Copy-Item .env.example .env
   ```

5. **Run the bot:**

   ```powershell
   .\run_bot.ps1
   ```

   This checks Ollama is running first, then launches the bot and tees output to `bot.log`.

## Hardware Requirements

- **Minimum:** 12 GB VRAM
- **Recommended:** RTX 3090 Ti (24 GB VRAM) for simultaneous multi-modal agency

## License

MIT — see [LICENSE](LICENSE).
