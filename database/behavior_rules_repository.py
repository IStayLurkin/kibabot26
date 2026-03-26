from database.db_connection import get_db


async def init_behavior_rules_db():
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS behavior_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_text TEXT NOT NULL UNIQUE,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_by TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.commit()


async def list_behavior_rules():
    db = await get_db()
    cursor = await db.execute("""
        SELECT id, rule_text, enabled, created_by, created_at, updated_at
        FROM behavior_rules
        ORDER BY id ASC
    """)
    rows = await cursor.fetchall()
    return [
        {
            "id": row[0],
            "rule_text": row[1],
            "enabled": bool(row[2]),
            "created_by": row[3] or "",
            "created_at": row[4],
            "updated_at": row[5],
        }
        for row in rows
    ]


async def add_behavior_rule(rule_text: str, created_by: str = ""):
    db = await get_db()
    await db.execute("""
        INSERT INTO behavior_rules (rule_text, enabled, created_by, updated_at)
        VALUES (?, 1, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(rule_text)
        DO UPDATE SET
            enabled = 1,
            created_by = CASE
                WHEN excluded.created_by != '' THEN excluded.created_by
                ELSE behavior_rules.created_by
            END,
            updated_at = CURRENT_TIMESTAMP
    """, (rule_text, created_by))
    await db.commit()


async def remove_behavior_rule(rule_id: int):
    db = await get_db()
    cursor = await db.execute("""
        DELETE FROM behavior_rules
        WHERE id = ?
    """, (rule_id,))
    await db.commit()
    return cursor.rowcount


async def update_behavior_rule(rule_id: int, rule_text: str):
    db = await get_db()
    cursor = await db.execute("""
        UPDATE behavior_rules
        SET rule_text = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (rule_text, rule_id))
    await db.commit()
    return cursor.rowcount


async def replace_behavior_rule(old_rule_text: str, new_rule_text: str, created_by: str = ""):
    db = await get_db()
    cursor = await db.execute("""
        UPDATE behavior_rules
        SET rule_text = ?, created_by = ?, updated_at = CURRENT_TIMESTAMP
        WHERE lower(rule_text) = lower(?)
    """, (new_rule_text, created_by, old_rule_text))
    await db.commit()
    return cursor.rowcount


async def clear_behavior_rules():
    db = await get_db()
    await db.execute("DELETE FROM behavior_rules")
    await db.commit()


async def init_bot_config_db():
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS bot_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    await db.commit()


async def get_bot_config(key: str, default: str = "") -> str:
    db = await get_db()
    cursor = await db.execute("SELECT value FROM bot_config WHERE key = ?", (key,))
    row = await cursor.fetchone()
    return row[0] if row else default


async def set_bot_config(key: str, value: str) -> None:
    db = await get_db()
    await db.execute(
        "INSERT INTO bot_config (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    await db.commit()
