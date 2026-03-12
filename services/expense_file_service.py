import asyncio
import json
import os


def build_export_data(rows):
    export_data = []

    for expense_id, date, category, amount, method, note in rows:
        export_data.append({
            "id": expense_id,
            "date": date,
            "category": category,
            "amount": amount,
            "method": method,
            "note": note,
        })

    return export_data


def write_export_file(rows, export_file: str = "expenses_export.json") -> str:
    export_data = build_export_data(rows)

    with open(export_file, "w", encoding="utf-8") as file:
        json.dump(export_data, file, indent=4)

    return export_file


async def write_export_file_async(rows, export_file: str = "expenses_export.json") -> str:
    return await asyncio.to_thread(write_export_file, rows, export_file)


def load_import_file(import_file: str = "expenses_import.json"):
    if not os.path.exists(import_file):
        return False, "No expenses_import.json file found.", None

    with open(import_file, "r", encoding="utf-8") as file:
        imported_expenses = json.load(file)

    if not isinstance(imported_expenses, list):
        return False, "Invalid format in expenses_import.json. Expected a list of expenses.", None

    return True, None, imported_expenses


async def load_import_file_async(import_file: str = "expenses_import.json"):
    return await asyncio.to_thread(load_import_file, import_file)


def normalize_imported_expenses(imported_expenses):
    normalized = []

    for expense in imported_expenses:
        if not isinstance(expense, dict):
            continue

        date = expense.get("date")
        category = expense.get("category")
        amount = expense.get("amount")
        method = expense.get("method", "Imported")
        note = expense.get("note", "")

        if not date or not category or amount is None:
            continue

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            continue

        normalized.append({
            "date": date,
            "category": str(category).lower(),
            "amount": amount,
            "method": method,
            "note": note,
        })

    return normalized
