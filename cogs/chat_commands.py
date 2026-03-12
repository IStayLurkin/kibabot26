import asyncio
import time

import discord
from discord.ext import commands

from core.constants import (
    BOT_ALLOWED_CHAT_CHANNELS,
    CHAT_COOLDOWN_SECONDS,
)
from database.chat_memory import (
    add_chat_message,
    get_or_create_session,
)
from services.chat_service import ChatReply, generate_dynamic_reply
from services.llm_service import LLMService
from services.memory_service import store_memory_if_found
from services.summary_service import maybe_update_summary
from services.tool_router import ToolRouter


class ChatCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.user_cooldowns = {}
        self.llm = getattr(bot, "llm_service", None) or LLMService()
        self.allowed_chat_channels = BOT_ALLOWED_CHAT_CHANNELS
        self.tool_router = ToolRouter()

    def is_on_cooldown(self, user_id: int, seconds: float = CHAT_COOLDOWN_SECONDS) -> bool:
        now = time.monotonic()
        last_used = self.user_cooldowns.get(user_id, 0.0)

        if now - last_used < seconds:
            return True

        self.user_cooldowns[user_id] = now
        return False

    def is_dm(self, message: discord.Message) -> bool:
        return isinstance(message.channel, discord.DMChannel)

    def is_allowed_chat_channel(self, message: discord.Message) -> bool:
        if self.is_dm(message):
            return True

        if not hasattr(message.channel, "name"):
            return False

        return message.channel.name.lower() in self.allowed_chat_channels

    def _build_chat_services(self) -> dict:
        return {
            "image_service": getattr(self.bot, "image_service", None),
            "voice_service": getattr(self.bot, "voice_service", None),
            "video_service": getattr(self.bot, "video_service", None),
            "codegen_service": getattr(self.bot, "codegen_service", None),
            "osint_service": getattr(self.bot, "osint_service", None),
            "model_runtime_service": getattr(self.bot, "model_runtime_service", None),
            "command_help_service": getattr(self.bot, "command_help_service", None),
            "bot": self.bot,
        }

    def get_typing_delay(self, content: str) -> float:
        cleaned = (content or "").strip()
        if not cleaned:
            return 0.35

        estimated_seconds = len(cleaned) / 90.0
        return min(5.0, max(0.35, estimated_seconds))

    async def send_chat_message(self, destination, reply):
        if isinstance(reply, str):
            async with destination.typing():
                await asyncio.sleep(self.get_typing_delay(reply))
            await destination.send(reply)
            return

        if not isinstance(reply, ChatReply):
            text_reply = str(reply)
            async with destination.typing():
                await asyncio.sleep(self.get_typing_delay(text_reply))
            await destination.send(text_reply)
            return

        files = [discord.File(path) for path in reply.file_paths if path]
        typing_delay = self.get_typing_delay(reply.content)

        async with destination.typing():
            await asyncio.sleep(typing_delay)

        if files:
            await destination.send(reply.content, files=files)
            return

        await destination.send(reply.content)

    async def maybe_send_tool_ack(self, destination, content: str):
        route_decision = self.tool_router.route(content)
        acknowledgements = {
            "image": "On it, generating that now...",
            "voice": "On it, making that audio now...",
            "video": "On it, starting that video request now...",
        }

        ack_message = acknowledgements.get(route_decision.tool_name)
        if ack_message:
            await destination.send(ack_message)

    async def handle_chat_turn(self, destination, author, channel, content: str):
        user_id = str(author.id)
        channel_id = str(channel.id)
        session_id = await get_or_create_session(user_id, channel_id)

        await add_chat_message(session_id, "user", content)

        stored_memory = await store_memory_if_found(self.llm, user_id, content)
        if stored_memory:
            key, value = stored_memory
            reply_text = f"Got it. I'll remember that {key} is `{value}`."
            await add_chat_message(session_id, "bot", reply_text)
            await self.send_chat_message(destination, reply_text)
            await maybe_update_summary(self.llm, user_id, channel_id, session_id)
            return

        await self.maybe_send_tool_ack(destination, content)

        reply = await generate_dynamic_reply(
            self.llm,
            display_name=author.display_name,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            user_text=content,
            services=self._build_chat_services(),
        )

        await add_chat_message(session_id, "bot", reply.content)
        await self.send_chat_message(destination, reply)
        await maybe_update_summary(self.llm, user_id, channel_id, session_id)

    @commands.command(aliases=["latency"])
    async def ping(self, ctx):
        embed = discord.Embed(
            title="Pong",
            description=f"Latency: `{round(self.bot.latency * 1000)}ms`",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @commands.command()
    async def about(self, ctx):
        runtime_service = getattr(self.bot, "model_runtime_service", None)
        provider = self.llm.provider
        model_name = self.llm._get_active_model_name()
        if runtime_service is not None:
            provider = runtime_service.get_active_llm_provider()
            model_name = runtime_service.get_active_llm_model()

        embed = discord.Embed(
            title="About Kiba Bot",
            description="A modular Discord bot with expense tracking and chat features.",
            color=discord.Color.blurple()
        )
        embed.add_field(name="Prefix", value="`!`", inline=True)
        embed.add_field(name="Mode", value="Prefix commands + agentic chat", inline=True)
        embed.add_field(name="Provider", value=f"`{provider}`", inline=True)
        embed.add_field(name="Model", value=f"`{model_name}`", inline=True)
        embed.add_field(
            name="Features",
            value="Expense tracking, memory, summaries, intent-aware chat, lightweight planning, tool routing, and media/code helpers",
            inline=False
        )
        await ctx.send(embed=embed)

    @commands.command(aliases=["ask", "talk"])
    async def chat(self, ctx, *, message: str):
        await self.handle_chat_turn(ctx, ctx.author, ctx.channel, message)

    @commands.command(aliases=["chathelp"])
    async def helpchat(self, ctx):
        embed = discord.Embed(
            title="Chat Commands",
            description="Available chat-related commands.",
            color=discord.Color.blurple()
        )
        embed.add_field(name="Ping", value="`!ping`", inline=True)
        embed.add_field(name="About", value="`!about`", inline=True)
        embed.add_field(name="Chat", value="`!chat <message>`", inline=False)
        embed.add_field(
            name="Examples",
            value=(
                "`!chat hello`\n"
                "`!chat help me plan a monthly budget`\n"
                "`!chat my image prompt is a neon fox in the rain`\n"
                "`!chat debug this python error`\n"
                "`!chat whois openai.com`\n"
                "`!chat what do you remember`\n"
                "`!chat what day is today`"
            ),
            inline=False
        )
        embed.add_field(
            name="Passive Replies",
            value=(
                "DMs: freeform messages work automatically.\n"
                "Servers: the bot replies when mentioned or when you reply to it in allowed channels."
            ),
            inline=False
        )
        await ctx.send(embed=embed)

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot:
            return

        if self.bot.user is None:
            return

        if message.content.startswith("!"):
            return

        is_dm = self.is_dm(message)
        is_mentioned = self.bot.user in message.mentions

        is_reply_to_bot = False
        if message.reference and message.reference.resolved:
            referenced_message = message.reference.resolved
            if referenced_message and referenced_message.author == self.bot.user:
                is_reply_to_bot = True

        if not is_dm and not is_mentioned and not is_reply_to_bot:
            return

        if not self.is_allowed_chat_channel(message):
            return

        if self.is_on_cooldown(message.author.id):
            return

        content = message.content.strip()

        if is_mentioned:
            mention_strings = [
                self.bot.user.mention,
                f"<@!{self.bot.user.id}>",
                f"<@{self.bot.user.id}>",
            ]
            for mention in mention_strings:
                content = content.replace(mention, "").strip()

        if not content:
            user_id = str(message.author.id)
            channel_id = str(message.channel.id)
            session_id = await get_or_create_session(user_id, channel_id)

            if is_dm:
                reply = (
                    f"Hey {message.author.display_name}, you can just talk to me here. "
                    "Tell me what you want to get done and I’ll help."
                )
            else:
                reply = (
                    f"Hey {message.author.display_name}, tell me what you're trying to do. "
                    "I can answer, plan, troubleshoot, or use tools when it helps."
                )

            await add_chat_message(session_id, "bot", reply)
            await self.send_chat_message(message.channel, reply)
            await maybe_update_summary(self.llm, user_id, channel_id, session_id)
            return

        await self.handle_chat_turn(message.channel, message.author, message.channel, content)


async def setup(bot):
    await bot.add_cog(ChatCommands(bot))
