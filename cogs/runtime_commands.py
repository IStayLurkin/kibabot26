from __future__ import annotations

from discord.ext import commands


class RuntimeCommands(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.runtime = getattr(bot, "model_runtime_service", None)
        self.help_service = getattr(bot, "command_help_service", None)

    @commands.group(name="model", invoke_without_command=True, help="Switch or inspect the active LLM model at runtime.")
    async def model_group(self, ctx: commands.Context):
        await ctx.send("Use `!model current`, `!model list`, `!model set <model>`, `!model sync`, or `!model add <provider> <model>`.")

    @model_group.command(name="current", help="Show the active LLM provider and model.")
    async def model_current(self, ctx: commands.Context):
        await ctx.send(self.runtime.get_current_model_text("llm"))

    @model_group.command(name="list", help="List registered LLM models.")
    async def model_list(self, ctx: commands.Context):
        await ctx.send(await self.runtime.get_model_list_text("llm"))

    @model_group.command(name="set", help="Switch the active LLM model at runtime.")
    async def model_set(self, ctx: commands.Context, *, model_name: str):
        _ok, message = await self.runtime.set_active_model("llm", model_name)
        await ctx.send(message)

    @model_group.command(name="sync", help="Discover LLM models from supported providers and local storage.")
    async def model_sync(self, ctx: commands.Context):
        result = await self.runtime.sync_models("llm")
        await ctx.send(f"LLM sync complete. Discovered {result['count']} models.")

    @model_group.command(name="add", help="Manually register an LLM model.")
    async def model_add(self, ctx: commands.Context, provider: str, *, model_name: str):
        await self.runtime.add_model(provider, model_name, "llm")
        await ctx.send(f"Registered LLM model `{provider}:{model_name}`.")

    @commands.group(name="imagemodel", invoke_without_command=True, help="Switch or inspect the active image model at runtime.")
    async def image_model_group(self, ctx: commands.Context):
        await ctx.send("Use `!imagemodel current`, `!imagemodel list`, `!imagemodel set <model>`, `!imagemodel sync`, or `!imagemodel add <provider> <model>`.")

    @image_model_group.command(name="current", help="Show the active image provider and model.")
    async def image_model_current(self, ctx: commands.Context):
        await ctx.send(self.runtime.get_current_model_text("image"))

    @image_model_group.command(name="list", help="List registered image models.")
    async def image_model_list(self, ctx: commands.Context):
        await ctx.send(await self.runtime.get_model_list_text("image"))

    @image_model_group.command(name="set", help="Switch the active image model at runtime.")
    async def image_model_set(self, ctx: commands.Context, *, model_name: str):
        _ok, message = await self.runtime.set_active_model("image", model_name)
        await ctx.send(message)

    @image_model_group.command(name="sync", help="Discover image models from supported providers and local storage.")
    async def image_model_sync(self, ctx: commands.Context):
        result = await self.runtime.sync_models("image")
        await ctx.send(f"Image sync complete. Discovered {result['count']} models.")

    @image_model_group.command(name="add", help="Manually register an image model.")
    async def image_model_add(self, ctx: commands.Context, provider: str, *, model_name: str):
        await self.runtime.add_model(provider, model_name, "image")
        await ctx.send(f"Registered image model `{provider}:{model_name}`.")

    @commands.command(name="cuda", help="Show CUDA and GPU status.")
    async def cuda_command(self, ctx: commands.Context, action: str | None = None):
        if action and action.strip().lower() not in {"status"}:
            await ctx.send("Usage: `!cuda` or `!cuda status`")
            return
        await ctx.send(await self.runtime.get_hardware_status_text())

    @commands.command(name="gpu", help="Show GPU, CUDA, and device status.")
    async def gpu_command(self, ctx: commands.Context, action: str | None = None):
        if action and action.strip().lower() not in {"status"}:
            await ctx.send("Usage: `!gpu` or `!gpu status`")
            return
        await ctx.send(await self.runtime.get_hardware_status_text())

    @commands.command(name="commands", aliases=["cmds"], help="List the commands currently available on the bot.")
    async def commands_list(self, ctx: commands.Context):
        await ctx.send(await self.help_service.build_command_overview(self.bot, ctx))

    @commands.command(name="help", help="Show dynamic help for all commands or a specific command.")
    async def help_command(self, ctx: commands.Context, *, command_name: str | None = None):
        if command_name:
            await ctx.send(await self.help_service.build_command_help(self.bot, command_name, ctx))
            return
        await ctx.send(await self.help_service.build_command_overview(self.bot, ctx))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(RuntimeCommands(bot))
