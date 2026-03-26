from __future__ import annotations

from discord.ext import commands

from core.config import CODE_MAX_OUTPUT_CHARS
from core.logging_config import get_logger

logger = get_logger(__name__)


class CodeCommands(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.code_service = getattr(bot, "code_execution_service", None)

    async def ensure_authorized(self, ctx: commands.Context) -> bool:
        if self.code_service is None:
            await ctx.send("Code execution service is not available.")
            return False

        if self.code_service.user_is_allowed(ctx.author):
            return True

        await ctx.send("You are not allowed to use code execution commands.")
        return False

    @commands.group(name="code", invoke_without_command=True, help="Manage sandboxed workspace files and run Python code safely.")
    async def code_group(self, ctx: commands.Context):
        if not await self.ensure_authorized(ctx):
            return
        await ctx.send(
            "Use `!code create`, `!code edit`, `!code read`, `!code run`, `!code list`, `!code delete`, `!code output`, or `!code ask`."
        )

    @code_group.command(name="create", help="Create a file inside the sandbox workspace.")
    async def code_create(self, ctx: commands.Context, filename: str, *, content: str):
        if not await self.ensure_authorized(ctx):
            return
        if "/" in filename or "\\" in filename:
            await ctx.send("Filename must not contain path separators. Use a plain filename like `script.py`.")
            return
        try:
            relative_path = self.code_service.create_file(filename, content)
            await ctx.send(f"Created `{relative_path}` in the sandbox workspace.")
        except Exception as exc:
            await ctx.send(f"Create failed: {exc}")

    @code_group.command(name="edit", help="Overwrite a file inside the sandbox workspace.")
    async def code_edit(self, ctx: commands.Context, filename: str, *, content: str):
        if not await self.ensure_authorized(ctx):
            return
        try:
            relative_path = self.code_service.edit_file(filename, content)
            await ctx.send(f"Updated `{relative_path}` in the sandbox workspace.")
        except Exception as exc:
            await ctx.send(f"Edit failed: {exc}")

    @code_group.command(name="read", help="Read a file from the sandbox workspace.")
    async def code_read(self, ctx: commands.Context, filename: str):
        if not await self.ensure_authorized(ctx):
            return
        try:
            content = self.code_service.read_file(filename)
        except Exception as exc:
            await ctx.send(f"Read failed: {exc}")
            return

        read_limit = min(CODE_MAX_OUTPUT_CHARS, 3800)
        if len(content) > read_limit:
            content = content[:read_limit] + "\n...[truncated]"
        await ctx.send(f"```python\n{content}\n```")

    @code_group.command(name="list", help="List files in the sandbox workspace.")
    async def code_list(self, ctx: commands.Context):
        if not await self.ensure_authorized(ctx):
            return
        files = self.code_service.list_files()
        if not files:
            await ctx.send("The sandbox workspace is empty.")
            return
        await ctx.send("Sandbox workspace files:\n" + "\n".join(f"- `{item}`" for item in files))

    @code_group.command(name="delete", help="Delete a file from the sandbox workspace.")
    async def code_delete(self, ctx: commands.Context, filename: str):
        if not await self.ensure_authorized(ctx):
            return
        try:
            relative_path = self.code_service.delete_file(filename)
            await ctx.send(f"Deleted `{relative_path}` from the sandbox workspace.")
        except Exception as exc:
            await ctx.send(f"Delete failed: {exc}")

    @code_group.command(name="run", help="Run a Python file inside the sandbox workspace.")
    @commands.cooldown(rate=3, per=60, type=commands.BucketType.user)
    async def code_run(self, ctx: commands.Context, filename: str, allow_flag: str | None = None):
        if not await self.ensure_authorized(ctx):
            return
        allow_dangerous = (allow_flag or "").strip().lower() == "--allow-dangerous"
        async with ctx.typing():
            try:
                result = await self.code_service.run_file(
                    filename,
                    user_id=str(ctx.author.id),
                    channel_id=str(ctx.channel.id),
                    allow_dangerous=allow_dangerous,
                )
            except Exception as exc:
                await ctx.send(f"Run failed: {exc}")
                return

        stdout_text = result["stdout_text"] or "(no stdout)"
        stderr_text = result["stderr_text"] or "(no stderr)"
        if len(stdout_text) > 1200:
            stdout_text = stdout_text[:1200] + "\n...[truncated]"
        if len(stderr_text) > 1200:
            stderr_text = stderr_text[:1200] + "\n...[truncated]"

        await ctx.send(
            f"Run ID: `{result['run_id']}`\n"
            f"File: `{result['filename']}`\n"
            f"Exit code: `{result['exit_code']}`\n"
            f"Duration: `{result['duration_ms']:.2f}ms`\n"
            f"Sandbox: `{result['sandbox_mode']}`\n"
            f"Stdout:\n```text\n{stdout_text}\n```\n"
            f"Stderr:\n```text\n{stderr_text}\n```"
        )

    @code_group.command(name="output", help="Fetch stored output for a previous sandbox run.")
    async def code_output(self, ctx: commands.Context, run_id: str):
        if not await self.ensure_authorized(ctx):
            return
        result = await self.code_service.get_run_output(run_id)
        if result is None:
            await ctx.send(f"I couldn't find a run with ID `{run_id}`.")
            return

        stdout_text = result["stdout_text"] or "(no stdout)"
        stderr_text = result["stderr_text"] or "(no stderr)"
        if len(stdout_text) > 1200:
            stdout_text = stdout_text[:1200] + "\n...[truncated]"
        if len(stderr_text) > 1200:
            stderr_text = stderr_text[:1200] + "\n...[truncated]"

        await ctx.send(
            f"Run ID: `{result['run_id']}`\n"
            f"File: `{result['filename']}`\n"
            f"Exit code: `{result['exit_code']}`\n"
            f"Created: `{result['created_at']}`\n"
            f"Stdout:\n```text\n{stdout_text}\n```\n"
            f"Stderr:\n```text\n{stderr_text}\n```"
        )


    @code_group.group(name="ask", invoke_without_command=True, help="Ask a coding model a question.")
    async def code_ask_group(self, ctx: commands.Context, *, prompt: str = ""):
        if not prompt:
            await ctx.send("Usage: `!code ask fast <prompt>` or `!code ask best <prompt>`")
            return
        await self._run_code_ask(ctx, prompt, "fast")

    @code_ask_group.command(name="fast", help="Ask the fast coding model.")
    async def code_ask_fast(self, ctx: commands.Context, *, prompt: str):
        await self._run_code_ask(ctx, prompt, "fast")

    @code_ask_group.command(name="best", help="Ask the best coding model.")
    async def code_ask_best(self, ctx: commands.Context, *, prompt: str):
        await self._run_code_ask(ctx, prompt, "best")

    async def _run_code_ask(self, ctx: commands.Context, prompt: str, tier: str):
        codegen = getattr(self.bot, "codegen_service", None)
        if codegen is None:
            await ctx.send("Codegen service is not available.")
            return
        async with ctx.typing():
            try:
                result = await codegen.ask(prompt, tier=tier)
                if len(result) <= 1900:
                    await ctx.send(result)
                else:
                    for i in range(0, len(result), 1900):
                        await ctx.send(result[i : i + 1900])
            except Exception as exc:
                logger.error("[code_ask] Error: %s", exc)
                await ctx.send(f"Code ask error: {exc}")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(CodeCommands(bot))
