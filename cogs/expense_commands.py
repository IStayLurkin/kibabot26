import discord
from discord.ext import commands

from core.constants import (
    EXPENSE_PER_PAGE,
    EXPENSE_RECENT_DEFAULT_COUNT,
)
from database.database import (
    add_expense,
    clear_expenses,
    delete_expense,
    get_all_expenses,
    get_category_totals,
    get_recent_expenses,
    get_total_expenses,
    search_expenses_by_category,
)
from services.expense_embed_service import (
    build_add_success_embed,
    build_categories_embed,
    build_clear_success_embed,
    build_dashboard_embed,
    build_delete_result_embed,
    build_help_embed,
    build_import_complete_embed,
    build_no_categories_embed,
    build_recent_embed,
    build_search_embed,
    build_start_embed,
    build_stats_embed,
    build_total_embed,
)
from services.expense_file_service import (
    load_import_file_async,
    normalize_imported_expenses,
    write_export_file_async,
)
from services.expense_validation_service import (
    normalize_category,
    validate_amount,
    validate_clear_confirmation,
    validate_recent_count,
)
from services.expense_view_service import ExpenseListView


class ExpenseCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(aliases=["expense", "menu"])
    async def start(self, ctx):
        await ctx.send(embed=build_start_embed())

    @commands.command(aliases=["home", "summary"])
    async def dashboard(self, ctx):
        total_amount = await get_total_expenses()
        category_rows = await get_category_totals()
        recent_rows = await get_recent_expenses(EXPENSE_RECENT_DEFAULT_COUNT)
        all_rows = await get_all_expenses()

        await ctx.send(
            embed=build_dashboard_embed(
                total_amount=total_amount,
                all_rows=all_rows,
                category_rows=category_rows,
                recent_rows=recent_rows,
            )
        )

    @commands.command(aliases=["a"])
    async def add(self, ctx, amount: float, category: str, *, note: str = ""):
        is_valid, error_message = validate_amount(amount)
        if not is_valid:
            await ctx.send(error_message)
            return

        category = normalize_category(category)
        date = ctx.message.created_at.strftime("%Y-%m-%d")

        await add_expense(date, category, amount, "Discord", note)

        await ctx.send(
            embed=build_add_success_embed(
                category=category,
                amount=amount,
                date=date,
                note=note,
            )
        )

    @commands.command(aliases=["tot"])
    async def total(self, ctx):
        total_amount = await get_total_expenses()
        await ctx.send(embed=build_total_embed(total_amount))

    @commands.command(aliases=["cats", "cat"])
    async def categories(self, ctx):
        rows = await get_category_totals()

        if not rows:
            await ctx.send(embed=build_no_categories_embed())
            return

        await ctx.send(embed=build_categories_embed(rows))

    @commands.command(name="expensehelp", aliases=["expense_help"])
    async def expense_help(self, ctx):
        await ctx.send(embed=build_help_embed())

    @commands.command(aliases=["del", "remove"])
    async def delete(self, ctx, expense_id: int):
        deleted_count = await delete_expense(expense_id)
        await ctx.send(embed=build_delete_result_embed(expense_id, deleted_count))

    @commands.command(aliases=["rec"])
    async def recent(self, ctx, count: int = EXPENSE_RECENT_DEFAULT_COUNT):
        is_valid, error_message = validate_recent_count(count)
        if not is_valid:
            await ctx.send(error_message)
            return

        rows = await get_recent_expenses(count)

        if not rows:
            await ctx.send("No expenses found.")
            return

        await ctx.send(embed=build_recent_embed(rows, count))

    @commands.command(name="list", aliases=["ls"])
    async def list_expenses(self, ctx):
        rows = await get_all_expenses()

        if not rows:
            await ctx.send("No expenses found.")
            return

        view = ExpenseListView(rows=rows, author_id=ctx.author.id, per_page=EXPENSE_PER_PAGE)
        message = await ctx.send(embed=view.build_embed(), view=view)
        view.message = message

    @commands.command(aliases=["clearall", "clr"])
    async def clear(self, ctx, confirm: str = ""):
        is_valid, error_message = validate_clear_confirmation(confirm)
        if not is_valid:
            await ctx.send(error_message)
            return

        await clear_expenses()
        await ctx.send(embed=build_clear_success_embed())

    @commands.command()
    async def export(self, ctx):
        rows = await get_all_expenses()

        if not rows:
            await ctx.send("No expenses to export.")
            return

        export_file = await write_export_file_async(rows)
        await ctx.send(file=discord.File(export_file))

    @commands.command(aliases=["imp"])
    async def import_expenses(self, ctx):
        ok, error_message, imported_expenses = await load_import_file_async()

        if not ok:
            await ctx.send(error_message)
            return

        normalized_expenses = normalize_imported_expenses(imported_expenses)
        imported_count = 0

        for expense in normalized_expenses:
            await add_expense(
                expense["date"],
                expense["category"],
                expense["amount"],
                expense["method"],
                expense["note"],
            )
            imported_count += 1

        await ctx.send(embed=build_import_complete_embed(imported_count))

    @commands.command(aliases=["statistics", "stat"])
    async def stats(self, ctx):
        total_amount = await get_total_expenses()
        category_rows = await get_category_totals()

        if total_amount == 0 and not category_rows:
            await ctx.send("No expenses found.")
            return

        await ctx.send(embed=build_stats_embed(total_amount, category_rows))

    @commands.command(aliases=["find"])
    async def search(self, ctx, category: str):
        category = normalize_category(category)
        rows = await search_expenses_by_category(category)

        if not rows:
            await ctx.send(f"No expenses found for category: {category}")
            return

        await ctx.send(embed=build_search_embed(rows, category))


async def setup(bot):
    await bot.add_cog(ExpenseCommands(bot))
