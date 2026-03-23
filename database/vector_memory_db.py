import struct
import logging
import aiosqlite
import sqlite_vec

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


async def init_vector_memory_db(db: aiosqlite.Connection) -> None:
    """Create vector_memories table. Call after loading sqlite_vec extension."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS vector_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_vector_memories_user
        ON vector_memories(user_id)
    """)
    await db.commit()


async def store_vector_memory(
    db: aiosqlite.Connection,
    user_id: str,
    content: str,
    embedding: list[float],
) -> None:
    """Store a memory with its embedding blob."""
    blob = _pack_embedding(embedding)
    await db.execute(
        "INSERT INTO vector_memories (user_id, content, embedding) VALUES (?, ?, ?)",
        (user_id, content, blob),
    )
    await db.commit()


async def get_all_vector_memories(
    db: aiosqlite.Connection,
    user_id: str,
) -> list[aiosqlite.Row]:
    """Return all stored memories for a user (content + embedding blob)."""
    cursor = await db.execute(
        "SELECT id, content, embedding, created_at FROM vector_memories WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    )
    return await cursor.fetchall()


async def delete_vector_memories(db: aiosqlite.Connection, user_id: str) -> None:
    """Delete all vector memories for a user."""
    await db.execute("DELETE FROM vector_memories WHERE user_id = ?", (user_id,))
    await db.commit()
