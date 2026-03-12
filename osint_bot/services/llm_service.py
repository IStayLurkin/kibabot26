from __future__ import annotations

import asyncio

from openai import OpenAI

from osint_bot.core.config import (
    OSINT_HF_BASE_URL,
    OSINT_HF_FALLBACK_ENABLED,
    OSINT_HF_MODEL,
    OSINT_HF_TOKEN,
    OSINT_LLM_PROVIDER,
    OSINT_OLLAMA_API_KEY,
    OSINT_OLLAMA_BASE_URL,
    OSINT_OLLAMA_MODEL,
)


class OSINTLLMService:
    def __init__(self) -> None:
        self.provider = OSINT_LLM_PROVIDER

    def get_active_model_name(self) -> str:
        if self.provider == "hf":
            return OSINT_HF_MODEL
        return OSINT_OLLAMA_MODEL

    def _client_for_provider(self, provider: str) -> tuple[OpenAI, str]:
        if provider == "hf":
            return (
                OpenAI(base_url=OSINT_HF_BASE_URL, api_key=OSINT_HF_TOKEN),
                OSINT_HF_MODEL,
            )
        return (
            OpenAI(base_url=OSINT_OLLAMA_BASE_URL, api_key=OSINT_OLLAMA_API_KEY),
            OSINT_OLLAMA_MODEL,
        )

    def _provider_chain(self) -> list[str]:
        if self.provider == "hf":
            return ["hf", "ollama"]
        if OSINT_HF_FALLBACK_ENABLED:
            return ["ollama", "hf"]
        return ["ollama"]

    def summarize_findings_sync(self, prompt: str) -> str:
        errors: list[str] = []
        for provider in self._provider_chain():
            try:
                client, model = self._client_for_provider(provider)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You summarize safe OSINT findings for Discord users. "
                                "Be concise, factual, and include no tactical abuse advice."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=260,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                errors.append(f"{provider}: empty response")
            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")

        raise RuntimeError("All LLM providers failed | " + " | ".join(errors))

    async def summarize_findings(self, prompt: str) -> str:
        return await asyncio.to_thread(self.summarize_findings_sync, prompt)
