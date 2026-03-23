import logging
import httpx

logger = logging.getLogger(__name__)

_EMBED_TIMEOUT = 10.0


class EmbeddingService:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns list of floats, or [] on failure."""
        try:
            async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
        except Exception as exc:
            logger.warning("[embedding] Failed to embed text: %s", exc)
            return []

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially. Returns list of embeddings (empty list on failure)."""
        results = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
        return results
