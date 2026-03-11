import discord
from core.constants import EXPENSE_LIST_TIMEOUT


class ExpenseListView(discord.ui.View):
    def __init__(self, rows, author_id: int, per_page: int = 10, timeout: float = EXPENSE_LIST_TIMEOUT):
        super().__init__(timeout=timeout)
        self.rows = rows
        self.author_id = author_id
        self.per_page = per_page
        self.page = 0
        self.message = None
        self._update_buttons()

    def max_page(self) -> int:
        if not self.rows:
            return 0
        return (len(self.rows) - 1) // self.per_page

    def page_slice(self):
        start = self.page * self.per_page
        end = start + self.per_page
        return self.rows[start:end]

    def build_embed(self) -> discord.Embed:
        embed = discord.Embed(
            title=f"Expense List — Page {self.page + 1}/{self.max_page() + 1}",
            color=discord.Color.blurple()
        )

        for expense_id, date, category, amount, method, note in self.page_slice():
            embed.add_field(
                name=f"ID {expense_id} • {category.title()} • ${amount:.2f}",
                value=f"Date: {date}\nMethod: {method}\nNote: {note or 'None'}",
                inline=False
            )

        embed.set_footer(text="Use the buttons below to change pages.")
        return embed

    def _update_buttons(self):
        self.prev_button.disabled = self.page <= 0
        self.next_button.disabled = self.page >= self.max_page()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "Only the person who ran this command can use these buttons.",
                ephemeral=True
            )
            return False
        return True

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True

        if self.message is not None:
            await self.message.edit(view=self)

    @discord.ui.button(label="Previous", style=discord.ButtonStyle.secondary)
    async def prev_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.page -= 1
        self._update_buttons()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.primary)
    async def next_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.page += 1
        self._update_buttons()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)