import json

from database.db_connection import get_db


async def init_model_registry_db():
    db = await get_db()

    # Check if table exists
    cursor = await db.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'model_registry'
    """)
    existing_table = await cursor.fetchone()

    # Check if already on current schema (has 'backend' column)
    already_migrated = False
    if existing_table:
        cursor = await db.execute("PRAGMA table_info(model_registry)")
        columns = {row[1] for row in await cursor.fetchall()}
        already_migrated = "backend" in columns

    if not already_migrated:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS model_registry_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL CHECK(model_type IN ('llm', 'image', 'audio')),
                source TEXT NOT NULL DEFAULT 'manual',
                enabled INTEGER NOT NULL DEFAULT 1,
                local_path TEXT,
                capabilities TEXT NOT NULL DEFAULT '[]',
                backend TEXT NOT NULL DEFAULT '',
                preferred_device TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_synced_at TEXT
            )
        """)

        if existing_table:
            try:
                await db.execute("""
                    INSERT INTO model_registry_new (
                        id,
                        provider,
                        model_name,
                        model_type,
                        source,
                        enabled,
                        local_path,
                        capabilities,
                        backend,
                        preferred_device,
                        created_at,
                        updated_at,
                        last_synced_at
                    )
                    SELECT
                        id,
                        provider,
                        model_name,
                        model_type,
                        source,
                        enabled,
                        local_path,
                        capabilities,
                        backend,
                        preferred_device,
                        created_at,
                        updated_at,
                        last_synced_at
                    FROM model_registry
                    WHERE model_type IN ('llm', 'image', 'audio')
                    ON CONFLICT(id) DO NOTHING
                """)
                await db.execute("DROP TABLE IF EXISTS model_registry")
                await db.execute("ALTER TABLE model_registry_new RENAME TO model_registry")
                await db.commit()
            except Exception:
                await db.execute("DROP TABLE IF EXISTS model_registry_new")
                await db.commit()
                raise
        else:
            await db.execute("ALTER TABLE model_registry_new RENAME TO model_registry")
            await db.commit()

    await db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_model_registry_unique
        ON model_registry(provider, model_name, model_type)
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS runtime_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    await db.commit()


async def upsert_model(
    provider: str,
    model_name: str,
    model_type: str,
    *,
    source: str = "manual",
    enabled: bool = True,
    local_path: str | None = None,
    capabilities: list[str] | None = None,
    backend: str = "",
    preferred_device: str = "",
    update_last_synced: bool = False,
):
    capabilities_json = json.dumps(capabilities or [])

    db = await get_db()
    await db.execute("""
        INSERT INTO model_registry (
            provider,
            model_name,
            model_type,
            source,
            enabled,
            local_path,
            capabilities,
            backend,
            preferred_device,
            updated_at,
            last_synced_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END)
        ON CONFLICT(provider, model_name, model_type)
        DO UPDATE SET
            source = excluded.source,
            enabled = excluded.enabled,
            local_path = excluded.local_path,
            capabilities = excluded.capabilities,
            backend = excluded.backend,
            preferred_device = excluded.preferred_device,
            updated_at = CURRENT_TIMESTAMP,
            last_synced_at = CASE
                WHEN ? THEN CURRENT_TIMESTAMP
                ELSE model_registry.last_synced_at
            END
    """, (
        provider,
        model_name,
        model_type,
        source,
        1 if enabled else 0,
        local_path,
        capabilities_json,
        backend,
        preferred_device,
        1 if update_last_synced else 0,
        1 if update_last_synced else 0,
    ))
    await db.commit()


async def list_models(model_type: str):
    db = await get_db()
    cursor = await db.execute("""
        SELECT
            provider,
            model_name,
            model_type,
            source,
            enabled,
            local_path,
            capabilities,
            backend,
            preferred_device,
            created_at,
            updated_at,
            last_synced_at
        FROM model_registry
        WHERE model_type = ?
        ORDER BY provider ASC, model_name ASC
    """, (model_type,))
    rows = await cursor.fetchall()

    models = []
    for row in rows:
        capabilities = []
        if row[6]:
            try:
                capabilities = json.loads(row[6])
            except json.JSONDecodeError:
                capabilities = []

        models.append({
            "provider": row[0],
            "model_name": row[1],
            "model_type": row[2],
            "source": row[3],
            "enabled": bool(row[4]),
            "local_path": row[5] or "",
            "capabilities": capabilities,
            "backend": row[7] or "",
            "preferred_device": row[8] or "",
            "created_at": row[9],
            "updated_at": row[10],
            "last_synced_at": row[11] or "",
        })

    return models


async def find_models(model_type: str, model_name: str):
    db = await get_db()
    cursor = await db.execute("""
        SELECT provider, model_name, model_type, source, enabled, local_path, capabilities, backend, preferred_device
        FROM model_registry
        WHERE model_type = ? AND lower(model_name) = lower(?)
        ORDER BY provider ASC
    """, (model_type, model_name))
    rows = await cursor.fetchall()

    results = []
    for row in rows:
        capabilities = []
        if row[6]:
            try:
                capabilities = json.loads(row[6])
            except json.JSONDecodeError:
                capabilities = []

        results.append({
            "provider": row[0],
            "model_name": row[1],
            "model_type": row[2],
            "source": row[3],
            "enabled": bool(row[4]),
            "local_path": row[5] or "",
            "capabilities": capabilities,
            "backend": row[7] or "",
            "preferred_device": row[8] or "",
        })

    return results


async def get_model(provider: str, model_name: str, model_type: str):
    db = await get_db()
    cursor = await db.execute("""
        SELECT provider, model_name, model_type, source, enabled, local_path, capabilities, backend, preferred_device
        FROM model_registry
        WHERE provider = ? AND model_name = ? AND model_type = ?
    """, (provider, model_name, model_type))
    row = await cursor.fetchone()

    if not row:
        return None

    capabilities = []
    if row[6]:
        try:
            capabilities = json.loads(row[6])
        except json.JSONDecodeError:
            capabilities = []

    return {
        "provider": row[0],
        "model_name": row[1],
        "model_type": row[2],
        "source": row[3],
        "enabled": bool(row[4]),
        "local_path": row[5] or "",
        "capabilities": capabilities,
        "backend": row[7] or "",
        "preferred_device": row[8] or "",
    }


async def set_runtime_setting(setting_key: str, setting_value: str):
    db = await get_db()
    await db.execute("""
        INSERT INTO runtime_settings (setting_key, setting_value, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(setting_key)
        DO UPDATE SET
            setting_value = excluded.setting_value,
            updated_at = CURRENT_TIMESTAMP
    """, (setting_key, setting_value))
    await db.commit()


async def get_runtime_settings():
    db = await get_db()
    cursor = await db.execute("""
        SELECT setting_key, setting_value
        FROM runtime_settings
        ORDER BY setting_key ASC
    """)
    rows = await cursor.fetchall()

    return {key: value for key, value in rows}
