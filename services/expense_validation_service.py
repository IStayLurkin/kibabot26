from core.constants import EXPENSE_RECENT_MAX_COUNT


def normalize_category(category: str) -> str:
    return category.strip().lower()


def validate_amount(amount: float):
    if amount <= 0:
        return False, "Amount must be greater than zero."
    return True, None


def validate_recent_count(count: int):
    if count <= 0:
        return False, "Count must be greater than 0."
    if count > EXPENSE_RECENT_MAX_COUNT:
        return False, f"Count must be {EXPENSE_RECENT_MAX_COUNT} or fewer expenses."
    return True, None


def validate_clear_confirmation(confirm: str):
    if confirm.lower() != "yes":
        return False, "Type `!clear yes` to confirm clearing all expenses."
    return True, None