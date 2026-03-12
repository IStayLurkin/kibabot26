# OSINT Bot

Isolated Discord sub-bot for safe OSINT workflows inside this repository.

## Scope

- Public-source enrichment only for usernames, domains, URLs, IPs, and user-supplied text
- Owned or authorized asset checks only for live infrastructure probes such as DNS, RDAP, SSL, and HTTP headers
- No private data gathering, credential use, covert collection, phishing, malware, or intrusive automation

## Run

1. Copy `osint_bot/.env.example` values into your root `.env` or OS environment.
2. Start Ollama if you want local summarization.
3. Launch with:

```powershell
python -m osint_bot.bot
```

## Commands

- `/osint domain`
- `/osint url`
- `/osint ip`
- `/osint username`
- `/osint summarize`
- `/osint policy`

Prefix equivalents use `!osint`.
