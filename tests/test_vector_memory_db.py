import pytest
import pytest_asyncio
import aiosqlite
import sqlite_vec
import struct
from database.vector_memory_db import init_vector_memory_db, store_vector_memory, get_all_vector_memories


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.enable_load_extension(True)
    await conn.load_extension(sqlite_vec.loadable_path())
    await init_vector_memory_db(conn)
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_store_and_retrieve_memory(db):
    embedding = [0.1] * 768
    await store_vector_memory(db, user_id="123", content="Brandon builds Discord bots", embedding=embedding)
    rows = await get_all_vector_memories(db, user_id="123")
    assert len(rows) == 1
    assert rows[0]["content"] == "Brandon builds Discord bots"


@pytest.mark.asyncio
async def test_store_multiple_memories(db):
    embedding = [0.1] * 768
    await store_vector_memory(db, user_id="123", content="Memory one", embedding=embedding)
    await store_vector_memory(db, user_id="123", content="Memory two", embedding=embedding)
    rows = await get_all_vector_memories(db, user_id="123")
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_memories_isolated_by_user(db):
    embedding = [0.1] * 768
    await store_vector_memory(db, user_id="AAA", content="User A memory", embedding=embedding)
    await store_vector_memory(db, user_id="BBB", content="User B memory", embedding=embedding)
    rows_a = await get_all_vector_memories(db, user_id="AAA")
    rows_b = await get_all_vector_memories(db, user_id="BBB")
    assert len(rows_a) == 1
    assert len(rows_b) == 1
    assert rows_a[0]["content"] == "User A memory"
