from __future__ import annotations

import asyncio
from core.config import MEM0_API_KEY
from core.logging_config import get_logger

logger = get_logger(__name__)


class Mem0Service:
    """
    Memory backend using Mem0. Implements the same store/retrieve interface as VectorMemoryService
    so chat_service.py can swap between them transparently.
    """

    def __init__(self):
        try:
            from mem0 import Memory
            config = {}
            if MEM0_API_KEY:
                config["api_key"] = MEM0_API_KEY
            self._mem = Memory.from_config(config) if config else Memory()
            logger.info("[mem0] Mem0 memory backend initialized")
        except ImportError:
            raise ImportError("mem0ai is not installed. Run: pip install mem0ai")

    async def store(self, db, user_id: str, content: str) -> None:
        """Store a memory for a user. db parameter is ignored (Mem0 manages its own storage)."""
        try:
            await asyncio.to_thread(self._mem.add, content, user_id=user_id)
            logger.info("[mem0] Stored memory for user %s: %r", user_id, content[:80])
        except Exception as exc:
            logger.warning("[mem0] Store failed for user %s: %s", user_id, exc)

    async def retrieve(self, db, user_id: str, query: str) -> list[str]:
        """Retrieve relevant memories. db parameter is ignored."""
        try:
            results = await asyncio.to_thread(self._mem.search, query, user_id=user_id, limit=5)
            memories = []
            for r in results:
                if isinstance(r, dict):
                    memories.append(r.get("memory", r.get("text", str(r))))
                else:
                    memories.append(str(r))
            return memories
        except Exception as exc:
            logger.warning("[mem0] Retrieve failed for user %s: %s", user_id, exc)
            return []

    async def get_all(self, user_id: str) -> list[str]:
        """Get all memories for a user."""
        try:
            results = await asyncio.to_thread(self._mem.get_all, user_id=user_id)
            return [r.get("memory", str(r)) if isinstance(r, dict) else str(r) for r in results]
        except Exception as exc:
            logger.warning("[mem0] get_all failed: %s", exc)
            return []
