from database.db_connection import get_db
from database.behavior_rules_repository import init_behavior_rules_db, init_bot_config_db
from database.chat_memory import init_chat_memory_db
from database.budget_repository import init_budget_table
from database.execution_repository import init_execution_db
from database.model_registry import init_model_registry_db


async def init_db():
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            method TEXT NOT NULL,
            note TEXT
        )
    """)
    await db.commit()

    await init_chat_memory_db()
    await init_budget_table()
    await init_behavior_rules_db()
    await init_model_registry_db()
    await init_execution_db()
    await init_bot_config_db()


async def get_category_totals_for_month(month_prefix: str):
    db = await get_db()
    cursor = await db.execute("""
        SELECT category, COALESCE(SUM(amount), 0)
        FROM expenses
        WHERE date LIKE ?
        GROUP BY category
        ORDER BY SUM(amount) DESC
    """, (f"{month_prefix}%",))
    return await cursor.fetchall()


async def add_expense(date, category, amount, method, note):
    db = await get_db()
    await db.execute("""
        INSERT INTO expenses (date, category, amount, method, note)
        VALUES (?, ?, ?, ?, ?)
    """, (date, category, amount, method, note))
    await db.commit()


async def get_all_expenses():
    db = await get_db()
    async with db.execute("""
        SELECT id, date, category, amount, method, note
        FROM expenses
        ORDER BY id ASC
    """) as cursor:
        return await cursor.fetchall()


async def delete_expense(expense_id):
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM expenses WHERE id = ?",
        (expense_id,)
    )
    await db.commit()
    return cursor.rowcount


async def get_total_expenses():
    db = await get_db()
    async with db.execute("SELECT SUM(amount) FROM expenses") as cursor:
        row = await cursor.fetchone()
        return row[0] or 0


async def get_category_totals():
    db = await get_db()
    async with db.execute("""
        SELECT category, SUM(amount)
        FROM expenses
        GROUP BY category
        ORDER BY category ASC
    """) as cursor:
        return await cursor.fetchall()


async def get_recent_expenses(count):
    db = await get_db()
    async with db.execute("""
        SELECT id, date, category, amount, method, note
        FROM expenses
        ORDER BY id DESC
        LIMIT ?
    """, (count,)) as cursor:
        return await cursor.fetchall()


async def search_expenses_by_category(category):
    db = await get_db()
    async with db.execute("""
        SELECT id, date, category, amount, method, note
        FROM expenses
        WHERE category = ?
        ORDER BY id DESC
    """, (category,)) as cursor:
        return await cursor.fetchall()


async def clear_expenses():
    db = await get_db()
    await db.execute("DELETE FROM expenses")
    await db.commit()
