from __future__ import annotations

from collections import defaultdict

from discord.ext import commands

from core.constants import BOT_DEFAULT_PREFIX

SECTION_LABELS = {
    "RuntimeCommands": "Model Controls",
    "MediaCommands": "Image Controls",
    "AgentCommands": "System",
    "ChatCommands": "General",
    "ExpenseCommands": "Expenses",
    "BudgetCommands": "Budgets",
}

DEV_COGS = {"DevCommands"}


class CommandHelpService:
    def _is_dev_only(self, command: commands.Command) -> bool:
        return bool(getattr(command, "hidden", False) or command.cog_name in DEV_COGS)

    async def _can_show_command(self, command: commands.Command, ctx: commands.Context | None) -> bool:
        if self._is_dev_only(command):
            if ctx is None:
                return False
            try:
                return await command.can_run(ctx)
            except Exception:
                return False

        if ctx is None:
            return True

        try:
            return await command.can_run(ctx)
        except Exception:
            return False

    def _section_name(self, command: commands.Command) -> str:
        return SECTION_LABELS.get(command.cog_name or "", command.cog_name or "General")

    def _format_command_line(self, command: commands.Command, prefix: str) -> str:
        if command.signature:
            return f"{prefix}{command.qualified_name} {command.signature}".strip()
        return f"{prefix}{command.qualified_name}"

    async def build_command_overview(self, bot: commands.Bot, ctx: commands.Context | None = None) -> str:
        grouped: dict[str, list[str]] = defaultdict(list)
        prefix = getattr(bot, "command_prefix", BOT_DEFAULT_PREFIX)
        if not isinstance(prefix, str):
            prefix = BOT_DEFAULT_PREFIX

        for command in sorted(bot.commands, key=lambda item: item.name):
            if isinstance(command, commands.Group):
                for subcommand in sorted(command.commands, key=lambda item: item.name):
                    if not await self._can_show_command(subcommand, ctx):
                        continue
                    grouped[self._section_name(subcommand)].append(self._format_command_line(subcommand, prefix))
                continue

            if not await self._can_show_command(command, ctx):
                continue
            grouped[self._section_name(command)].append(self._format_command_line(command, prefix))

        if not grouped:
            return "No commands are currently available."

        lines = ["**Available Commands**"]
        for section in sorted(grouped.keys()):
            lines.append("")
            lines.append(f"**{section}**")
            for command_line in grouped[section]:
                lines.append(command_line)

        return "\n".join(lines)

    async def build_command_help(self, bot: commands.Bot, command_name: str, ctx: commands.Context | None = None) -> str:
        prefix = getattr(bot, "command_prefix", BOT_DEFAULT_PREFIX)
        if not isinstance(prefix, str):
            prefix = BOT_DEFAULT_PREFIX

        normalized = command_name.strip().lstrip(prefix).strip()
        command = bot.get_command(normalized)
        if command is None:
            return f"I couldn't find a command named `{command_name}`."

        if not await self._can_show_command(command, ctx):
            return f"I couldn't find a command named `{command_name}`."

        lines = [f"**Command:** {prefix}{command.qualified_name}"]
        if command.help:
            lines.append(f"**Description:** {command.help}")
        elif command.brief:
            lines.append(f"**Description:** {command.brief}")

        lines.append(f"**Usage:** {self._format_command_line(command, prefix)}")

        aliases = [alias for alias in command.aliases if alias != command.name]
        if aliases:
            lines.append(f"**Aliases:** {', '.join(f'{prefix}{alias}' for alias in aliases)}")

        if isinstance(command, commands.Group) and command.commands:
            subcommands = ", ".join(f"{prefix}{subcommand.qualified_name}" for subcommand in sorted(command.commands, key=lambda item: item.name))
            lines.append(f"**Subcommands:** {subcommands}")

        return "\n".join(lines)

    def matches_natural_language_help(self, text: str) -> bool:
        lowered = text.strip().lower()
        triggers = (
            "what commands do you have",
            "what can you do",
            "show commands",
            "list commands",
            "help",
        )
        return lowered in triggers or any(trigger in lowered for trigger in triggers[:-1])
