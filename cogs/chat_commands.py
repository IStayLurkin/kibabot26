import asyncio
import time
import gc
import torch
import discord
import subprocess
import os
from discord.ext import commands

from core.constants import (
    BOT_ALLOWED_CHAT_CHANNELS,
    CHAT_COOLDOWN_SECONDS,
)
from database.chat_memory import (
    add_chat_message,
    get_or_create_session,
)
from services.llm_service import LLMService
from services.agent_dispatcher import AgentDispatcher
from services.summary_service import maybe_update_summary # Re-added for memory consistency

class ChatCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.user_cooldowns = {}
        # Force local LLM initialization
        self.llm = getattr(bot, "llm_service", None) or LLMService()
        self.allowed_chat_channels = BOT_ALLOWED_CHAT_CHANNELS
        # Initialize the LangGraph Dispatcher
        self.dispatcher = AgentDispatcher(bot)

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

    @commands.command(name="status", aliases=["kiba", "vram"])
    async def kiba_dashboard(self, ctx):
        """Displays the 3090 Ti Status Dashboard."""
        # 1. Get Hardware Stats
        used_vram = self._get_vram_usage()
        total_vram = 24576 # 3090 Ti Total
        vram_pct = round((used_vram / total_vram) * 100, 1)
        
        # 2. Identify Active Model
        active_engine = "Ollama (Qwen3-Coder)" # Default
        if used_vram > 12000:
            # Check service states
            img_svc = getattr(self.bot, "image_service", None)
            mus_svc = getattr(self.bot, "music_service", None)
            
            if img_svc and img_svc.pipeline:
                active_engine = "FLUX.2 [Primary Media]"
            elif mus_svc and mus_svc.active_model_type:
                active_engine = f"YuE Studio ({mus_svc.active_model_type})"

        # 3. Build the UI
        embed = discord.Embed(
            title="🐺 Kiba Local AI Dashboard",
            color=discord.Color.dark_theme(),
            timestamp=ctx.message.created_at
        )
        
        # VRAM Progress Bar logic
        bar_length = 15
        filled = int((used_vram / total_vram) * bar_length)
        vram_bar = "█" * filled + "░" * (bar_length - filled)
        
        embed.add_field(name="Current Active Engine", value=f"**{active_engine}**", inline=False)
        embed.add_field(name="VRAM Utilization", value=f"`{vram_bar}` {vram_pct}%", inline=False)
        embed.add_field(name="Used / Total", value=f"{used_vram} MB / {total_vram} MB", inline=True)
        
        status_color = "🟢 Stable" if vram_pct < 90 else "🔴 CRITICAL (OOM Risk)"
        embed.add_field(name="Neural Stability", value=status_color, inline=True)
        
        embed.set_footer(text="3090 Ti | Sequential Offloading Active")
        await ctx.send(embed=embed)

    @commands.command(name="hardware")    
    async def hardware_stats(self, ctx):
        """Shows real-time VRAM and Load for the 3090 Ti."""
        try:
            cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,nounits,noheader"
            result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            used, total, load = result.split(', ')

            embed = discord.Embed(
                title="🖥️ Hardware Monitor (3090 Ti)", 
                color=discord.Color.blue(),
                description="Real-time status of the local inference engine."
            )
            embed.add_field(name="VRAM Usage", value=f"**{used} MB** / {total} MB", inline=True)
            embed.add_field(name="GPU Load", value=f"**{load}%**", inline=True)
            embed.add_field(name="Status", value="🟢 Online / Unfiltered", inline=False)
            embed.set_footer(text="Running via local Ollama instance")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Could not retrieve GPU stats: {e}")

    @commands.command(name="boost")
    async def turbo_mode(self, ctx):
        """Manually clears GPU cache and system garbage collection."""
        initial_vram = self._get_vram_usage()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        final_vram = self._get_vram_usage()
        freed = initial_vram - final_vram
        embed = discord.Embed(
            title="🚀 Turbo Mode Activated",
            description=f"Cleared **{freed} MB** of VRAM.",
            color=discord.Color.gold()
        )
        await ctx.send(embed=embed)

    def _get_vram_usage(self):
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
        return int(subprocess.check_output(cmd, shell=True).decode().strip())

    async def handle_chat_turn(self, destination, author, channel, content: str):
        """Routes requests through LangGraph for hardware-aware agent dispatching."""
        user_id = str(author.id)
        channel_id = str(channel.id)
        session_id = await get_or_create_session(user_id, channel_id)

        await add_chat_message(session_id, "user", content)

        async with destination.typing():
            try:
                # 1. Dispatch through LangGraph (handles VRAM swapping internally)
                response_text, file_path = await self.dispatcher.run(user_id, content)
                
                # 2. Log and Store result
                if response_text:
                    await add_chat_message(session_id, "bot", response_text)
                    await destination.send(response_text)
                
                # 3. Handle File Uploads (FLUX.2 renders)
                if file_path and os.path.exists(file_path):
                    # Use unique filename to avoid Discord caching issues
                    filename = f"kiba_{int(time.time())}.png"
                    file = discord.File(file_path, filename=filename)
                    await destination.send(file=file)
                    print(f"[DEBUG] Sent FLUX.2 render to {author.display_name}")
                
                # 4. Update Memory Summary (Only for text turns)
                if not file_path:
                    await maybe_update_summary(self.llm, user_id, channel_id, session_id)

            except Exception as e:
                print(f"[ERROR] Dispatcher failed: {e}")
                await destination.send("❌ Kiba is experiencing a neural sync error. Check terminal.")

    async def handle_natural_chat(self, message: discord.Message):
        """Bridge for DM/Mention chat."""
        if message.author.bot:
            return

        content = message.content.strip()
        if self.bot.user in message.mentions:
            mention_strings = [f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"]
            for m in mention_strings:
                content = content.replace(m, "").strip()

        if not content:
            return

        await self.handle_chat_turn(message.channel, message.author, message.channel, content)

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
            description="3090 Ti Powered Local AI Hub.",
            color=discord.Color.blurple()
        )
        embed.add_field(name="LLM Provider", value="`Ollama (Qwen3-Coder)`")
        embed.add_field(name="Media Engine", value="`Diffusers (FLUX.2)`")
        await ctx.send(embed=embed)

    @commands.command(aliases=["ask", "talk"])
    async def chat(self, ctx, *, message: str):
        await self.handle_chat_turn(ctx, ctx.author, ctx.channel, message)

    @commands.command(name="studio")
    async def set_studio_config(self, ctx, setting: str, value: str):
        """
        Configures the 3090 Ti Studio Engine.
        Usage: !studio bpm 140 | !studio voice female | !studio mode lyrics
        """
        music_service = getattr(self.bot, "music_service", None)
        if not music_service:
            return await ctx.send("❌ Music Service not loaded.")

        setting = setting.lower()
        try:
            if setting == "bpm":
                music_service.update_studio_settings(bpm=int(value))
            elif setting == "voice":
                music_service.update_studio_settings(voice=value)
            elif setting == "mode":
                music_service.update_studio_settings(mode=value)
            else:
                return await ctx.send("❓ Unknown setting. Use: bpm, voice, or mode.")
                
            embed = discord.Embed(
                title="🎼 Studio Profile Updated",
                description=f"**{setting.upper()}** set to `{value}`",
                color=discord.Color.purple()
            )
            await ctx.send(embed=embed)
        except ValueError:
            await ctx.send("❌ Invalid value. BPM must be a number.")

async def setup(bot):
    await bot.add_cog(ChatCommands(bot))