import math
import struct
import logging

from database.vector_memory_db import store_vector_memory, get_all_vector_memories

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class VectorMemoryService:
    def __init__(self, embedding_service, top_k: int = 5):
        self._embed = embedding_service
        self._top_k = top_k

    async def store(self, db, user_id: str, content: str) -> None:
        """Embed content and store in vector_memories. Silently skips on embed failure."""
        embedding = await self._embed.embed(content)
        if not embedding:
            logger.warning("[vector_memory] Skipping store — embed returned empty for user %s", user_id)
            return
        await store_vector_memory(db, user_id=user_id, content=content, embedding=embedding)

    async def retrieve(self, db, user_id: str, query: str) -> list[str]:
        """Embed query and return top-K most similar memory contents. Returns [] on failure."""
        try:
            query_vec = await self._embed.embed(query)
            if not query_vec:
                return []
            rows = await get_all_vector_memories(db, user_id=user_id)
            if not rows:
                return []
            scored = []
            for row in rows:
                blob = row["embedding"]
                vec = _unpack_embedding(blob)
                score = _cosine_similarity(query_vec, vec)
                scored.append((score, row["content"]))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [content for _, content in scored[: self._top_k]]
        except Exception as exc:
            logger.warning("[vector_memory] Retrieval failed: %s", exc)
            return []
