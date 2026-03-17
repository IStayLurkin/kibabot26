import aiosqlite

DB_PATH = "bot.db"
_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    global _connection
    if _connection is None:
        _connection = await aiosqlite.connect(DB_PATH)
        await _connection.execute("PRAGMA journal_mode=WAL")
        await _connection.execute("PRAGMA foreign_keys = ON")
        await _connection.execute("PRAGMA synchronous = NORMAL")
        _connection.row_factory = aiosqlite.Row
    return _connection


async def close_db():
    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None
