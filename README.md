# Kiba Bot 🤖

Kiba is a high-performance, agentic Discord bot designed for **fully local execution**. Built on the **2026 local AI stack**, it leverages the **RTX 3090 Ti** to run unrestricted Large Language Models and high-fidelity image generation simultaneously.

## 🚀 Features
- **Unrestricted Local Intelligence:** Powered by a custom Ollama `kiba` model (Dolphin 3.0 base).
- **GPU-Accelerated Rendering:** Local image generation using Stable Diffusion XL (SDXL) via the `diffusers` library.
- **Persistent Memory:** Long-term and short-term memory stored locally to recall user facts and conversation context.
- **Hardware Optimized:** Specifically tuned for **24GB VRAM** environments and **CUDA 12.4+**.

## 🛠️ Tech Stack
- **Backend:** Python 3.12
- **LLM Engine:** Ollama (Dolphin 3.0 / Llama 3.1/3.2)
- **Frameworks:** Discord.py, PyTorch (CUDA 12.4), Hugging Face Transformers
- **Storage:** G: Drive Optimized (No C: drive bloat)

## 📦 Installation

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/BrandonTaylor/KibaBot.git](https://github.com/BrandonTaylor/KibaBot.git)
   cd KibaBot
1. **Environment Setup (PowerShell):**

PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Configure Ollama:

Ensure Ollama is installed and mapped to your high-capacity drive.

Run the included Modelfile to bake the unrestricted personality:

PowerShell
ollama create kiba -f kiba.modelfile
Configuration:

Rename .env.example to .env and fill in your Discord Token.

⚠️ Hardware Requirements
Minimum: 12GB VRAM

Recommended: RTX 3090 Ti (24GB VRAM) for simultaneous multi-modal agency.


---

### 3. The MIT License (`LICENSE`)
Create this file in your root folder: `G:\code\python\learn_python\bot\discord_bot_things\LICENSE`. 

```text
MIT License

Copyright (c) 2026 Brandon Taylor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.