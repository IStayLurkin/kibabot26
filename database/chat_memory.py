import aiosqlite

DB_PATH = "bot.db"


async def init_chat_memory_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, channel_id)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                author_type TEXT NOT NULL CHECK(author_type IN ('user', 'bot')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, memory_key)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, channel_id)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                goal TEXT NOT NULL DEFAULT '',
                last_intent TEXT NOT NULL DEFAULT '',
                response_mode TEXT NOT NULL DEFAULT '',
                last_tool TEXT NOT NULL DEFAULT '',
                pending_question TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, channel_id)
            )
        """)

        await db.commit()


async def get_or_create_session(user_id: str, channel_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")

        cursor = await db.execute("""
            SELECT id
            FROM chat_sessions
            WHERE user_id = ? AND channel_id = ?
        """, (user_id, channel_id))
        row = await cursor.fetchone()

        if row:
            return row[0]

        cursor = await db.execute("""
            INSERT INTO chat_sessions (user_id, channel_id)
            VALUES (?, ?)
        """, (user_id, channel_id))
        await db.commit()
        return cursor.lastrowid


async def get_conversation_summary(user_id: str, channel_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT summary
            FROM chat_summaries
            WHERE user_id = ? AND channel_id = ?
        """, (user_id, channel_id))
        row = await cursor.fetchone()
        return row[0] if row else ""


async def set_conversation_summary(user_id: str, channel_id: str, summary: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO chat_summaries (user_id, channel_id, summary, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, channel_id)
            DO UPDATE SET
                summary = excluded.summary,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, channel_id, summary))
        await db.commit()


async def add_chat_message(session_id: int, author_type: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")

        await db.execute("""
            INSERT INTO chat_messages (session_id, author_type, content)
            VALUES (?, ?, ?)
        """, (session_id, author_type, content))

        await db.execute("""
            UPDATE chat_sessions
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (session_id,))

        await db.commit()


async def get_recent_chat_messages(session_id: int, limit: int = 10):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT author_type, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit))
        rows = await cursor.fetchall()
        return list(reversed(rows))


async def set_user_memory(user_id: str, memory_key: str, memory_value: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO user_memory (user_id, memory_key, memory_value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, memory_key)
            DO UPDATE SET
                memory_value = excluded.memory_value,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, memory_key, memory_value))
        await db.commit()


async def get_user_memory(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT memory_key, memory_value
            FROM user_memory
            WHERE user_id = ?
            ORDER BY memory_key
        """, (user_id,))
        return await cursor.fetchall()


async def get_conversation_state(user_id: str, channel_id: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT goal, last_intent, response_mode, last_tool, pending_question
            FROM chat_state
            WHERE user_id = ? AND channel_id = ?
        """, (user_id, channel_id))
        row = await cursor.fetchone()

        if not row:
            return {
                "goal": "",
                "last_intent": "",
                "response_mode": "",
                "last_tool": "",
                "pending_question": "",
            }

        return {
            "goal": row[0] or "",
            "last_intent": row[1] or "",
            "response_mode": row[2] or "",
            "last_tool": row[3] or "",
            "pending_question": row[4] or "",
        }


async def set_conversation_state(
    user_id: str,
    channel_id: str,
    *,
    goal: str = "",
    last_intent: str = "",
    response_mode: str = "",
    last_tool: str = "",
    pending_question: str = "",
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO chat_state (
                user_id,
                channel_id,
                goal,
                last_intent,
                response_mode,
                last_tool,
                pending_question,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, channel_id)
            DO UPDATE SET
                goal = excluded.goal,
                last_intent = excluded.last_intent,
                response_mode = excluded.response_mode,
                last_tool = excluded.last_tool,
                pending_question = excluded.pending_question,
                updated_at = CURRENT_TIMESTAMP
        """, (
            user_id,
            channel_id,
            goal,
            last_intent,
            response_mode,
            last_tool,
            pending_question,
        ))
        await db.commit()
