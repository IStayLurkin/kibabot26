import time
import discord
from discord.ext import commands
from database.chat_memory import (
    get_or_create_session,
    add_chat_message,
)
from services.llm_service import LLMService
from services.memory_service import store_memory_if_found
from services.summary_service import maybe_update_summary
from services.chat_service import generate_dynamic_reply
from core.constants import (
    BOT_ALLOWED_CHAT_CHANNELS,
    CHAT_COOLDOWN_SECONDS,
)


class ChatCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.user_cooldowns = {}
        self.llm = getattr(bot, "llm_service", None) or LLMService()
        self.allowed_chat_channels = BOT_ALLOWED_CHAT_CHANNELS

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

    async def send_chat_message(self, destination, content: str):
        await destination.send(content)

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
        embed = discord.Embed(
            title="About Kiba Bot",
            description="A modular Discord bot with expense tracking and chat features.",
            color=discord.Color.blurple()
        )
        embed.add_field(name="Prefix", value="`!`", inline=True)
        embed.add_field(name="Mode", value="Prefix commands + freeform chat", inline=True)
        embed.add_field(name="Provider", value=f"`{self.llm.provider}`", inline=True)
        embed.add_field(name="Model", value=f"`{self.llm._get_active_model_name()}`", inline=True)
        embed.add_field(
            name="Features",
            value="Expense tracking, embeds, pagination, chat commands, DMs, freeform replies, memory, summaries, hot-swappable LLMs, live date/time",
            inline=False
        )
        await ctx.send(embed=embed)

    @commands.command(aliases=["ask", "talk"])
    async def chat(self, ctx, *, message: str):
        user_id = str(ctx.author.id)
        channel_id = str(ctx.channel.id)
        session_id = await get_or_create_session(user_id, channel_id)

        await add_chat_message(session_id, "user", message)

        stored_memory = await store_memory_if_found(self.llm, user_id, message)
        if stored_memory:
            key, value = stored_memory
            reply = f"Got it. I'll remember that {key} is `{value}`."
            await add_chat_message(session_id, "bot", reply)
            await self.send_chat_message(ctx, reply)
            await maybe_update_summary(self.llm, user_id, channel_id, session_id)
            return

        reply = await generate_dynamic_reply(
            self.llm,
            display_name=ctx.author.display_name,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            user_text=message,
        )

        await add_chat_message(session_id, "bot", reply)
        await self.send_chat_message(ctx, reply)
        await maybe_update_summary(self.llm, user_id, channel_id, session_id)

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
                "`!chat help`\n"
                "`!chat how do i add an expense`\n"
                "`!chat show my recent expenses`\n"
                "`!chat my name is Brandon`\n"
                "`!chat what do you remember`\n"
                "`!chat what were we talking about earlier`\n"
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

        user_id = str(message.author.id)
        channel_id = str(message.channel.id)
        session_id = await get_or_create_session(user_id, channel_id)

        if content:
            await add_chat_message(session_id, "user", content)

            stored_memory = await store_memory_if_found(self.llm, user_id, content)
            if stored_memory:
                key, value = stored_memory
                reply = f"Got it. I'll remember that {key} is `{value}`."
                await add_chat_message(session_id, "bot", reply)
                await self.send_chat_message(message.channel, reply)
                await maybe_update_summary(self.llm, user_id, channel_id, session_id)
                return

        if not content:
            if is_dm:
                reply = (
                    f"Hey {message.author.display_name}, you can just talk to me here. "
                    "Try asking how to add an expense or use `!help`."
                )
            else:
                reply = (
                    f"Hey {message.author.display_name}, try `!help`, `!helpchat`, "
                    "or `!chat <message>`."
                )

            await add_chat_message(session_id, "bot", reply)
            await self.send_chat_message(message.channel, reply)
            await maybe_update_summary(self.llm, user_id, channel_id, session_id)
            return

        reply = await generate_dynamic_reply(
            self.llm,
            display_name=message.author.display_name,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            user_text=content,
        )

        await add_chat_message(session_id, "bot", reply)
        await self.send_chat_message(message.channel, reply)
        await maybe_update_summary(self.llm, user_id, channel_id, session_id)


async def setup(bot):
    await bot.add_cog(ChatCommands(bot))