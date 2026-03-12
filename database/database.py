import aiosqlite
from database.chat_memory import init_chat_memory_db
from database.budget_repository import init_budget_table

DB_NAME = "bot.db"


async def init_db():
    async with aiosqlite.connect(DB_NAME) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                method TEXT NOT NULL,
                note TEXT
            )
        """)
        await conn.commit()

    await init_chat_memory_db()
    await init_budget_table()


async def get_category_totals_for_month(month_prefix: str):
    async with aiosqlite.connect(DB_NAME) as conn:
        cursor = await conn.execute("""
            SELECT category, COALESCE(SUM(amount), 0)
            FROM expenses
            WHERE date LIKE ?
            GROUP BY category
            ORDER BY SUM(amount) DESC
        """, (f"{month_prefix}%",))
        return await cursor.fetchall()


async def add_expense(date, category, amount, method, note):
    async with aiosqlite.connect(DB_NAME) as conn:
        await conn.execute("""
            INSERT INTO expenses (date, category, amount, method, note)
            VALUES (?, ?, ?, ?, ?)
        """, (date, category, amount, method, note))
        await conn.commit()


async def get_all_expenses():
    async with aiosqlite.connect(DB_NAME) as conn:
        async with conn.execute("""
            SELECT id, date, category, amount, method, note
            FROM expenses
            ORDER BY id ASC
        """) as cursor:
            return await cursor.fetchall()


async def delete_expense(expense_id):
    async with aiosqlite.connect(DB_NAME) as conn:
        cursor = await conn.execute(
            "DELETE FROM expenses WHERE id = ?",
            (expense_id,)
        )
        await conn.commit()
        return cursor.rowcount


async def get_total_expenses():
    async with aiosqlite.connect(DB_NAME) as conn:
        async with conn.execute("SELECT SUM(amount) FROM expenses") as cursor:
            row = await cursor.fetchone()
            return row[0] or 0


async def get_category_totals():
    async with aiosqlite.connect(DB_NAME) as conn:
        async with conn.execute("""
            SELECT category, SUM(amount)
            FROM expenses
            GROUP BY category
            ORDER BY category ASC
        """) as cursor:
            return await cursor.fetchall()


async def get_recent_expenses(count):
    async with aiosqlite.connect(DB_NAME) as conn:
        async with conn.execute("""
            SELECT id, date, category, amount, method, note
            FROM expenses
            ORDER BY id DESC
            LIMIT ?
        """, (count,)) as cursor:
            return await cursor.fetchall()


async def search_expenses_by_category(category):
    async with aiosqlite.connect(DB_NAME) as conn:
        async with conn.execute("""
            SELECT id, date, category, amount, method, note
            FROM expenses
            WHERE category = ?
            ORDER BY id DESC
        """, (category,)) as cursor:
            return await cursor.fetchall()


async def clear_expenses():
    async with aiosqlite.connect(DB_NAME) as conn:
        await conn.execute("DELETE FROM expenses")
        await conn.commit()
