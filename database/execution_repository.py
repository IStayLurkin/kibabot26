from database.db_connection import get_db


async def init_execution_db():
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS code_runs (
            run_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            command TEXT NOT NULL,
            exit_code INTEGER NOT NULL,
            duration_ms REAL NOT NULL,
            stdout_text TEXT NOT NULL DEFAULT '',
            stderr_text TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.commit()


async def add_code_run(
    run_id: str,
    user_id: str,
    channel_id: str,
    filename: str,
    command: str,
    exit_code: int,
    duration_ms: float,
    stdout_text: str,
    stderr_text: str,
):
    db = await get_db()
    await db.execute("""
        INSERT INTO code_runs (
            run_id,
            user_id,
            channel_id,
            filename,
            command,
            exit_code,
            duration_ms,
            stdout_text,
            stderr_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        user_id,
        channel_id,
        filename,
        command,
        exit_code,
        duration_ms,
        stdout_text,
        stderr_text,
    ))
    await db.commit()


async def get_code_run(run_id: str):
    db = await get_db()
    cursor = await db.execute("""
        SELECT
            run_id,
            user_id,
            channel_id,
            filename,
            command,
            exit_code,
            duration_ms,
            stdout_text,
            stderr_text,
            created_at
        FROM code_runs
        WHERE run_id = ?
    """, (run_id,))
    row = await cursor.fetchone()

    if not row:
        return None

    return {
        "run_id": row[0],
        "user_id": row[1],
        "channel_id": row[2],
        "filename": row[3],
        "command": row[4],
        "exit_code": row[5],
        "duration_ms": row[6],
        "stdout_text": row[7],
        "stderr_text": row[8],
        "created_at": row[9],
    }
