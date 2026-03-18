from database.db_connection import get_db


async def init_budget_table():
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY,
            category TEXT UNIQUE,
            amount REAL NOT NULL
        )
    """)
    await db.commit()


async def set_budget(category: str, amount: float):
    db = await get_db()
    await db.execute("""
        INSERT INTO budgets (category, amount)
        VALUES (?, ?)
        ON CONFLICT(category) DO UPDATE SET amount = excluded.amount
    """, (category, amount))
    await db.commit()


async def get_budgets():
    db = await get_db()
    cursor = await db.execute("""
        SELECT category, amount
        FROM budgets
        ORDER BY category ASC
    """)
    return await cursor.fetchall()


async def delete_budget(category: str):
    db = await get_db()
    await db.execute("DELETE FROM budgets WHERE category = ?", (category,))
    await db.commit()
