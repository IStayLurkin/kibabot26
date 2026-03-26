from __future__ import annotations

from discord.ext import commands
from database.behavior_rules_repository import get_bot_config, set_bot_config
from services.llm_service import PERSONALITIES, DEFAULT_PERSONALITY

_RUNTIME_UNAVAILABLE = "❌ Model runtime service is not available."


class RuntimeCommands(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.runtime = getattr(bot, "model_runtime_service", None)
        self.help_service = getattr(bot, "command_help_service", None)
        self.behavior_rule_service = getattr(bot, "behavior_rule_service", None)

    @commands.group(name="model", invoke_without_command=True, help="Switch or inspect the active LLM model at runtime.")
    async def model_group(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(
            self.runtime.get_current_model_text("llm")
            + "\n\nUse `!model list`, `!model current`, `!model switch <model>`, `!model pull <model>`, `!model reload`, `!model sync`, or `!model add <provider> <model>`."
        )

    @model_group.command(name="current", help="Show the active LLM provider and model.")
    async def model_current(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(self.runtime.get_current_model_text("llm"))

    @model_group.command(name="list", help="List registered LLM models.")
    async def model_list(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(await self.runtime.get_model_list_text("llm"))

    @model_group.command(name="set", help="Switch the active LLM model at runtime.")
    async def model_set(self, ctx: commands.Context, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        _ok, message = await self.runtime.set_active_model("llm", model_name)
        await ctx.send(message)

    @model_group.command(name="pull", help="Pull or install a model into local/provider-managed storage.")
    async def model_pull(self, ctx: commands.Context, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        _ok, message = await self.runtime.pull_model("llm", model_name)
        await ctx.send(message)

    @model_group.command(name="reload", help="Reload model discovery, runtime state, and storage-backed models.")
    async def model_reload(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(await self.runtime.reload_runtime_state())

    @model_group.command(name="sync", help="Discover LLM models from supported providers and local storage.")
    async def model_sync(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        result = await self.runtime.sync_models("llm")
        await ctx.send(f"LLM sync complete. Discovered {result['count']} models.")

    @model_group.command(name="add", help="Manually register an LLM model.")
    async def model_add(self, ctx: commands.Context, provider: str, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await self.runtime.add_model(provider, model_name, "llm")
        await ctx.send(f"Registered LLM model `{provider}:{model_name}`.")

    @commands.group(name="imagemodel", invoke_without_command=True, help="Switch or inspect the active image model at runtime.")
    async def image_model_group(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(
            self.runtime.get_current_model_text("image")
            + "\n\nUse `!imagemodel current`, `!imagemodel list`, `!imagemodel switch <model>`, `!imagemodel pull <model>`, `!imagemodel reload`, `!imagemodel sync`, or `!imagemodel add <provider> <model>`."
        )

    @image_model_group.command(name="current", help="Show the active image provider and model.")
    async def image_model_current(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(self.runtime.get_current_model_text("image"))

    @image_model_group.command(name="list", help="List registered image models.")
    async def image_model_list(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(await self.runtime.get_model_list_text("image"))

    @image_model_group.command(name="set", help="Switch the active image model at runtime.")
    async def image_model_set(self, ctx: commands.Context, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        _ok, message = await self.runtime.set_active_model("image", model_name)
        await ctx.send(message)

    @image_model_group.command(name="pull", help="Pull or install an image model into local/provider-managed storage.")
    async def image_model_pull(self, ctx: commands.Context, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        _ok, message = await self.runtime.pull_model("image", model_name)
        await ctx.send(message)

    @image_model_group.command(name="reload", help="Reload image model discovery and runtime state.")
    async def image_model_reload(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(await self.runtime.reload_runtime_state())

    @image_model_group.command(name="sync", help="Discover image models from supported providers and local storage.")
    async def image_model_sync(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        result = await self.runtime.sync_models("image")
        await ctx.send(f"Image sync complete. Discovered {result['count']} models.")

    @image_model_group.command(name="add", help="Manually register an image model.")
    async def image_model_add(self, ctx: commands.Context, provider: str, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await self.runtime.add_model(provider, model_name, "image")
        await ctx.send(f"Registered image model `{provider}:{model_name}`.")

    @commands.group(name="audiomodel", invoke_without_command=True, help="Switch or inspect the active audio/TTS model at runtime.")
    async def audio_model_group(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(
            self.runtime.get_current_model_text("audio")
            + "\n\nUse `!audiomodel current`, `!audiomodel list`, `!audiomodel switch <model>`, `!audiomodel pull <model>`, `!audiomodel reload`, `!audiomodel sync`, or `!audiomodel add <provider> <model>`."
        )

    @audio_model_group.command(name="current", help="Show the active audio provider and model.")
    async def audio_model_current(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(self.runtime.get_current_model_text("audio"))

    @audio_model_group.command(name="list", help="List registered audio models.")
    async def audio_model_list(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(await self.runtime.get_model_list_text("audio"))

    @audio_model_group.command(name="set", help="Switch the active audio model at runtime.")
    async def audio_model_set(self, ctx: commands.Context, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        _ok, message = await self.runtime.set_active_model("audio", model_name)
        await ctx.send(message)

    @audio_model_group.command(name="pull", help="Pull or install an audio model into local/provider-managed storage.")
    async def audio_model_pull(self, ctx: commands.Context, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        _ok, message = await self.runtime.pull_model("audio", model_name)
        await ctx.send(message)

    @audio_model_group.command(name="reload", help="Reload audio model discovery and runtime state.")
    async def audio_model_reload(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await ctx.send(await self.runtime.reload_runtime_state())

    @audio_model_group.command(name="sync", help="Discover audio models from supported providers.")
    async def audio_model_sync(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        result = await self.runtime.sync_models("audio")
        await ctx.send(f"Audio sync complete. Discovered {result['count']} models.")

    @audio_model_group.command(name="add", help="Manually register an audio model.")
    async def audio_model_add(self, ctx: commands.Context, provider: str, *, model_name: str):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
        await self.runtime.add_model(provider, model_name, "audio")
        await ctx.send(f"Registered audio model `{provider}:{model_name}`.")

    @commands.command(name="cuda", aliases=["gpu"], help="Show CUDA and GPU status.")
    async def cuda_command(self, ctx: commands.Context):
        if self.runtime is None:
            return await ctx.send(_RUNTIME_UNAVAILABLE)
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

    @commands.group(name="rule", invoke_without_command=True, help="Create, view, and manage strict persistent bot behavior rules.")
    async def rule_group(self, ctx: commands.Context):
        await ctx.send(
            await self.behavior_rule_service.get_rules_text()
            + "\n\nUse `!rule add <text>`, `!rule edit <id> <text>`, `!rule list`, `!rule delete <id>`, or `!rule clear`."
        )

    @rule_group.command(name="list", help="List all persistent behavior rules.")
    async def rule_list(self, ctx: commands.Context):
        await ctx.send(await self.behavior_rule_service.get_rules_text())

    @rule_group.command(name="add", aliases=["set", "create"], help="Add a strict persistent behavior rule for the bot.")
    async def rule_add(self, ctx: commands.Context, *, rule_text: str):
        _ok, message = await self.behavior_rule_service.add_rule(rule_text, created_by=str(ctx.author.id))
        await ctx.send(message)

    @rule_group.command(name="edit", help="Edit a behavior rule by ID.")
    async def rule_edit(self, ctx: commands.Context, rule_id: int, *, rule_text: str):
        _ok, message = await self.behavior_rule_service.edit_rule(rule_id, rule_text)
        await ctx.send(message)

    @rule_group.command(name="delete", aliases=["remove"], help="Delete a behavior rule by ID.")
    async def rule_delete(self, ctx: commands.Context, rule_id: int):
        _ok, message = await self.behavior_rule_service.delete_rule(rule_id)
        await ctx.send(message)

    @rule_group.command(name="clear", help="Clear all custom behavior rules.")
    async def rule_clear(self, ctx: commands.Context):
        _ok, message = await self.behavior_rule_service.reset_rules()
        await ctx.send(message)


    @commands.group(name="personality", invoke_without_command=True, help="Switch or view Kiba's personality for your conversations.")
    async def personality_group(self, ctx: commands.Context):
        user_key = f"user_personality:{ctx.author.id}"
        current = await get_bot_config(user_key, "")
        if not current or current not in PERSONALITIES:
            current = await get_bot_config("active_personality", DEFAULT_PERSONALITY)
        names = ", ".join(f"`{k}`" for k in PERSONALITIES)
        await ctx.send(f"Your personality: `{current}`\nAvailable: {names}\n\nUse `!personality set <name>` to switch yours. Use `!personality global <name>` to change the default for everyone.")

    @personality_group.command(name="list", help="List available personalities.")
    async def personality_list(self, ctx: commands.Context):
        lines = [f"`{name}` — {prompt.strip().splitlines()[1].strip()}" for name, prompt in PERSONALITIES.items()]
        await ctx.send("\n".join(lines))

    @personality_group.command(name="set", aliases=["switch"], help="Set your personal personality (only affects your conversations).")
    async def personality_set(self, ctx: commands.Context, *, name: str):
        name = name.lower().strip()
        if name not in PERSONALITIES:
            names = ", ".join(f"`{k}`" for k in PERSONALITIES)
            return await ctx.send(f"Unknown personality `{name}`. Available: {names}")
        await set_bot_config(f"user_personality:{ctx.author.id}", name)
        await ctx.send(f"Your personality set to `{name}`.")

    @personality_group.command(name="reset", help="Reset your personality to the server default.")
    async def personality_reset(self, ctx: commands.Context):
        await set_bot_config(f"user_personality:{ctx.author.id}", "")
        default = await get_bot_config("active_personality", DEFAULT_PERSONALITY)
        await ctx.send(f"Reset to server default: `{default}`.")

    @personality_group.command(name="global", help="Set the default personality for all users (owner only).")
    @commands.is_owner()
    async def personality_global(self, ctx: commands.Context, *, name: str):
        name = name.lower().strip()
        if name not in PERSONALITIES:
            names = ", ".join(f"`{k}`" for k in PERSONALITIES)
            return await ctx.send(f"Unknown personality `{name}`. Available: {names}")
        llm = getattr(self.bot, "llm_service", None)
        if llm is not None:
            llm.active_personality = name
        await set_bot_config("active_personality", name)
        await ctx.send(f"Global default personality set to `{name}`.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(RuntimeCommands(bot))
