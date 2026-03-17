import pytest
import asyncio
from database.db_connection import get_db, close_db, _connection

@pytest.mark.asyncio
async def test_get_db_returns_connection():
    db = await get_db()
    assert db is not None

@pytest.mark.asyncio
async def test_get_db_returns_same_instance():
    db1 = await get_db()
    db2 = await get_db()
    assert db1 is db2

@pytest.mark.asyncio
async def test_close_db_clears_connection():
    await get_db()
    await close_db()
    import database.db_connection as mod
    assert mod._connection is None
