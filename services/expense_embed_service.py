import discord


def build_start_embed():
    embed = discord.Embed(
        title="Expense Bot",
        description="Track expenses with simple prefix commands.",
        color=discord.Color.blurple()
    )

    embed.add_field(
        name="Quick Start",
        value=(
            "`!add 12.50 food lunch`\n"
            "`!list`\n"
            "`!recent 5`\n"
            "`!total`"
        ),
        inline=False
    )

    embed.add_field(
        name="Main Commands",
        value=(
            "`!add` `!total` `!categories` `!list`\n"
            "`!recent` `!search` `!delete` `!stats`\n"
            "`!clear yes` `!export` `!import_expenses`"
        ),
        inline=False
    )

    embed.set_footer(text="Use !help for the full command list.")
    return embed


def build_dashboard_embed(total_amount, all_rows, category_rows, recent_rows):
    embed = discord.Embed(
        title="Expense Dashboard",
        color=discord.Color.blurple()
    )

    embed.add_field(
        name="Total Spent",
        value=f"${total_amount:.2f}",
        inline=True
    )

    embed.add_field(
        name="Entries",
        value=str(len(all_rows)),
        inline=True
    )

    if category_rows:
        top_category, top_total = category_rows[0]
        embed.add_field(
            name="Top Category",
            value=f"{top_category.title()} (${top_total:.2f})",
            inline=True
        )
    else:
        embed.add_field(
            name="Top Category",
            value="None",
            inline=True
        )

    if recent_rows:
        recent_text = "\n".join(
            f"`#{expense_id}` {category.title()} - ${amount:.2f}"
            for expense_id, date, category, amount, method, note in recent_rows
        )
    else:
        recent_text = "No expenses recorded yet."

    embed.add_field(
        name="Recent Activity",
        value=recent_text,
        inline=False
    )

    return embed


def build_add_success_embed(category: str, amount: float, date: str, note: str):
    embed = discord.Embed(
        title="Expense Added",
        color=discord.Color.green()
    )
    embed.add_field(name="Category", value=category.title(), inline=True)
    embed.add_field(name="Amount", value=f"${amount:.2f}", inline=True)
    embed.add_field(name="Date", value=date, inline=True)
    embed.add_field(name="Note", value=note or "None", inline=False)
    return embed


def build_total_embed(total_amount: float):
    return discord.Embed(
        title="Total Expenses",
        description=f"**${total_amount:.2f}**",
        color=discord.Color.green()
    )


def build_no_categories_embed():
    return discord.Embed(
        title="Expenses by Category",
        description="No expenses found yet.",
        color=discord.Color.orange()
    )


def build_categories_embed(rows):
    embed = discord.Embed(
        title="Expenses by Category",
        color=discord.Color.gold()
    )

    for category, total in rows:
        embed.add_field(
            name=category.title(),
            value=f"${total:.2f}",
            inline=True
        )

    return embed


def build_help_embed():
    embed = discord.Embed(
        title="Expense Bot Commands",
        description="Available commands for managing expenses",
        color=discord.Color.blue()
    )

    embed.add_field(name="Start / Menu", value="`!start`", inline=True)
    embed.add_field(name="Dashboard", value="`!dashboard`", inline=True)
    embed.add_field(name="Add Expense", value="`!add <amount> <category> [note]`", inline=False)
    embed.add_field(name="Total", value="`!total`", inline=True)
    embed.add_field(name="Categories", value="`!categories`", inline=True)
    embed.add_field(name="List", value="`!list`", inline=True)
    embed.add_field(name="Recent", value="`!recent [count]`", inline=True)
    embed.add_field(name="Delete", value="`!delete <id>`", inline=True)
    embed.add_field(name="Search", value="`!search <category>`", inline=True)
    embed.add_field(name="Stats", value="`!stats`", inline=True)
    embed.add_field(name="Clear", value="`!clear yes`", inline=True)
    embed.add_field(name="Import / Export", value="`!import_expenses` / `!export`", inline=False)

    embed.set_footer(text="Aliases: !a, !tot, !cat, !ls, !del, !rec, !imp, !stat, !find")
    return embed


def build_delete_result_embed(expense_id: int, deleted_count: int):
    if deleted_count:
        return discord.Embed(
            title="Expense Deleted",
            description=f"Deleted expense with ID **{expense_id}**.",
            color=discord.Color.red()
        )

    return discord.Embed(
        title="Delete Failed",
        description=f"No expense found with ID **{expense_id}**.",
        color=discord.Color.orange()
    )


def build_recent_embed(rows, count: int):
    embed = discord.Embed(
        title=f"Recent Expenses ({count})",
        color=discord.Color.orange()
    )

    for expense_id, date, category, amount, method, note in rows:
        embed.add_field(
            name=f"ID {expense_id} • {category.title()} • ${amount:.2f}",
            value=f"Date: {date}\nMethod: {method}\nNote: {note or 'None'}",
            inline=False
        )

    return embed


def build_clear_success_embed():
    return discord.Embed(
        title="All Expenses Cleared",
        color=discord.Color.red()
    )


def build_import_complete_embed(imported_count: int):
    return discord.Embed(
        title="Import Complete",
        description=f"Imported **{imported_count}** expenses from `expenses_import.json`.",
        color=discord.Color.green()
    )


def build_stats_embed(total_amount: float, category_rows):
    embed = discord.Embed(
        title="Expense Statistics",
        color=discord.Color.purple()
    )

    embed.add_field(
        name="Total Expenses",
        value=f"${total_amount:.2f}",
        inline=False
    )

    category_text = ""
    for category, total in category_rows:
        category_text += f"**{category.title()}**: ${total:.2f}\n"

    embed.add_field(
        name="By Category",
        value=category_text or "No category data",
        inline=False
    )

    return embed


def build_search_embed(rows, category: str):
    embed = discord.Embed(
        title=f"Expenses in Category: {category.title()}",
        color=discord.Color.teal()
    )

    for expense_id, date, db_category, amount, method, note in rows:
        embed.add_field(
            name=f"ID {expense_id} • ${amount:.2f}",
            value=f"Date: {date}\nMethod: {method}\nNote: {note or 'None'}",
            inline=False
        )

    return embed
