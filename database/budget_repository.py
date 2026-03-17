import aiosqlite

DB_PATH = "bot.db"


async def init_budget_table():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY,
                category TEXT UNIQUE,
                amount REAL NOT NULL
            )
        """)
        await db.commit()


async def set_budget(category: str, amount: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO budgets (category, amount)
            VALUES (?, ?)
            ON CONFLICT(category) DO UPDATE SET amount = excluded.amount
        """, (category, amount))
        await db.commit()


async def get_budgets():
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT category, amount
            FROM budgets
            ORDER BY category ASC
        """)
        rows = await cursor.fetchall()
        return rows


async def delete_budget(category: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM budgets WHERE category = ?", (category,))
        await db.commit()
