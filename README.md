# Kiba Bot

Discord bot with:

- prefix commands only
- expense tracking
- SQLite + aiosqlite
- modular cogs/services/database architecture
- AI chat with memory and summaries
- OpenAI / Hugging Face / Ollama support
- live runtime date and time handling

## Setup

Create and activate a venv:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## OSINT Sub-Bot

An isolated OSINT Discord sub-bot lives in `osint_bot/`.
Run it separately with:

```powershell
python -m osint_bot.bot
```

See `osint_bot/README.md` for config and command details.
