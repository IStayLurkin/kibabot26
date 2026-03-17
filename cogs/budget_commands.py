import os, json, time
import os
from datetime import datetime
import discord
from discord.ext import commands

from database.budget_repository import set_budget, get_budgets, delete_budget
from database.database import get_category_totals_for_month
from services.expense_validation_service import normalize_category, validate_amount


class BudgetCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.group(invoke_without_command=True)
    async def budget(self, ctx: commands.Context):
        embed = discord.Embed(
            title="Budget Commands",
            description=(
                "`!budget set <category> <amount>`\n"
                "`!budget status`\n"
                "`!budget delete <category>`"
            ),
            color=discord.Color.blurple()
        )
        await ctx.send(embed=embed)

    @budget.command(name="set")
    async def budget_set(self, ctx: commands.Context, category: str, amount: float):
        category = normalize_category(category)
        is_valid, error_message = validate_amount(amount)
        if not is_valid:
            await ctx.send(error_message)
            return

        await set_budget(category, amount)

        embed = discord.Embed(
            title="Budget Updated",
            description=f"Set **{category}** budget to `${amount:.2f}`",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @budget.command(name="delete")
    async def budget_delete(self, ctx: commands.Context, category: str):
        category = normalize_category(category)
        await delete_budget(category)

        embed = discord.Embed(
            title="Budget Deleted",
            description=f"Removed budget for **{category}**",
            color=discord.Color.orange()
        )
        await ctx.send(embed=embed)

    @budget.command(name="status")
    async def budget_status(self, ctx: commands.Context):
        month_prefix = datetime.now().strftime("%Y-%m")
        budgets = await get_budgets()
        spent_rows = await get_category_totals_for_month(month_prefix)

        spent_map = {category: total for category, total in spent_rows}

        if not budgets:
            embed = discord.Embed(
                title="Budget Status",
                description="No budgets have been set yet.",
                color=discord.Color.blurple()
            )
            await ctx.send(embed=embed)
            return

        embed = discord.Embed(
            title="Budget Status",
            description=f"Tracking for **{month_prefix}**",
            color=discord.Color.blurple()
        )

        for category, limit_amount in budgets:
            spent = spent_map.get(category, 0)
            remaining = limit_amount - spent
            percent = (spent / limit_amount * 100) if limit_amount > 0 else 0

            if percent >= 100:
                status = "Over budget"
            elif percent >= 80:
                status = "Near limit"
            else:
                status = "On track"

            embed.add_field(
                name=category.title(),
                value=(
                    f"Spent: `${spent:.2f}`\n"
                    f"Budget: `${limit_amount:.2f}`\n"
                    f"Remaining: `${remaining:.2f}`\n"
                    f"Usage: `{percent:.1f}%` — {status}"
                ),
                inline=False
            )

        await ctx.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(BudgetCommands(bot))
