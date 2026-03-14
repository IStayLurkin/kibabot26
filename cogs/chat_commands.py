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
from services.summary_service import maybe_update_summary 

class ChatCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.user_cooldowns = {}
        # Force local LLM initialization
        self.llm = getattr(bot, "llm_service", None) or LLMService()
        self.allowed_chat_channels = BOT_ALLOWED_CHAT_CHANNELS
        # Initialize the LangGraph Dispatcher
        self.dispatcher = AgentDispatcher(bot)
        # Reference to the image service for easy access
        self.image_service = getattr(bot, "image_service", None)

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

    @commands.command(name="status", aliases=["kiba", "kb"])
    async def kiba_dashboard(self, ctx):
        """Displays the 3090 Ti Status Dashboard."""
        used_vram = self._get_vram_usage()
        total_vram = 24576 
        vram_pct = round((used_vram / total_vram) * 100, 1)
        
        active_engine = "Ollama (Qwen3-Coder)"
        if used_vram > 12000:
            img_svc = self.image_service
            mus_svc = getattr(self.bot, "music_service", None)
            
            if img_svc and img_svc.pipeline:
                # Identify if we are in FLUX or SDXL mode
                engine_name = getattr(img_svc, "current_engine", "FLUX.2")
                active_engine = f"{engine_name} [Active Media]"
            elif mus_svc and mus_svc.active_model_type:
                active_engine = f"YuE Studio ({mus_svc.active_model_type})"

        embed = discord.Embed(
            title="🐺 Kiba Local AI Dashboard",
            color=discord.Color.dark_theme(),
            timestamp=ctx.message.created_at
        )
        
        bar_length = 15
        filled = int((used_vram / total_vram) * bar_length)
        vram_bar = "█" * filled + "░" * (bar_length - filled)
        
        embed.add_field(name="Current Active Engine", value=f"**{active_engine}**", inline=False)
        embed.add_field(name="VRAM Utilization", value=f"`{vram_bar}` {vram_pct}%", inline=False)
        embed.add_field(name="Used / Total", value=f"{used_vram} MB / {total_vram} MB", inline=True)
        
        status_color = "🟢 Stable" if vram_pct < 90 else "🔴 CRITICAL (OOM Risk)"
        embed.add_field(name="Neural Stability", value=status_color, inline=True)
        
        embed.set_footer(text="3090 Ti | Multi-Engine Hot-Swapping Active")
        await ctx.send(embed=embed)

    @commands.command(name="hardware")    
    async def hardware_stats(self, ctx):
        """Shows real-time VRAM and Load for the 3090 Ti."""
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,nounits,noheader"]
            result = subprocess.check_output(cmd, shell=False).decode('utf-8').strip()
            used, total, load = result.split(', ')

            embed = discord.Embed(
                title="🖥️ Hardware Monitor (3090 Ti)", 
                color=discord.Color.blue(),
                description="Real-time status of the local inference engine."
            )
            embed.add_field(name="VRAM Usage", value=f"**{used} MB** / {total} MB", inline=True)
            embed.add_field(name="GPU Load", value=f"**{load}%**", inline=True)
            embed.add_field(name="Status", value="🟢 Online / Unfiltered", inline=False)
            embed.set_footer(text="Running via local hardware")
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

    @commands.command(name="dossier", aliases=["intel", "research"])
    async def dossier(self, ctx, target: str):
        """Triggers the new 2026 Agentic OSINT Research Loop."""
        osint_svc = getattr(self.bot, "osint_service", None)
        if not osint_svc:
            return await ctx.send("❌ OSINT Service not active.")
            
        async with ctx.typing():
            report = await osint_svc.run_dossier(target)
            await ctx.send(report)

    # --- IMAGE GENERATION COMMANDS ---

    @commands.command(name="draw")
    async def draw_flux(self, ctx, *, prompt: str):
        """Generates High-Quality imagery using FLUX.2 (Slow/Detailed)."""
        await self.handle_image_request(ctx, prompt, mode="FLUX")

    @commands.command(name="fast", aliases=["quick"])
    async def draw_sdxl(self, ctx, *, prompt: str):
        """Generates High-Speed imagery using SDXL (Fast/Stylized)."""
        await self.handle_image_request(ctx, prompt, mode="SDXL")

    async def handle_image_request(self, ctx, prompt: str, mode: str = "FLUX"):
        gallery_channel_id = 1482242041755861032
        icon = "🎨" if mode == "FLUX" else "⚡"
        
        status_msg = await ctx.send(f"{icon} **Kiba is initializing {mode} on the 3090 Ti...**\n[░░░░░░░░░░] 0%")

        def update_bar(percent, vram_gb):
            blocks = int(percent / 10)
            bar = "█" * blocks + "░" * (10 - blocks)
            asyncio.run_coroutine_threadsafe(
                status_msg.edit(content=(
                    f"{icon} **Kiba is rendering ({mode})...**\n"
                    f"[{bar}] {percent}%\n"
                    f"📟 **VRAM:** {vram_gb}GB / 24.0GB"
                )),
                self.bot.loop
            )

        # Call the unified service logic
        if mode == "SDXL":
            path = await self.image_service.generate_sdxl(prompt, progress_callback=update_bar)
        else:
            path = await self.image_service.generate_image(prompt, progress_callback=update_bar)

        if path:
            await status_msg.edit(content=f"✅ **{mode} Generation Complete!**")
            
            image_file = discord.File(path, filename=f"kiba_{mode.lower()}.png")
            await ctx.send(content=f"Request: *{prompt}* ({mode} Engine)", file=image_file)

            # Archive to Gallery
            gallery_channel = self.bot.get_channel(gallery_channel_id)
            if gallery_channel:
                archive_file = discord.File(path, filename=f"archive_{mode.lower()}.png")
                await gallery_channel.send(
                    content=f"🖼️ **Engine:** {mode}\n👤 **User:** {ctx.author.mention}\n📝 **Prompt:** {prompt}",
                    file=archive_file
                )
        else:
            await status_msg.edit(content=f"❌ **{mode} Engine Error.** Check terminal.")

    # --- CORE CHAT LOGIC ---

    def _get_vram_usage(self):
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        return int(subprocess.check_output(cmd, shell=False).decode().strip())

    async def handle_chat_turn(self, destination, author, channel, content: str):
        user_id = str(author.id)
        channel_id = str(channel.id)
        session_id = await get_or_create_session(user_id, channel_id)

        attachments = getattr(destination, "attachments", [])
        if attachments:
            for attachment in attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.ogg', '.m4a']):
                    voice_svc = getattr(self.bot, "voice_service", None)
                    if voice_svc:
                        temp_path = f"temp_{attachment.filename}"
                        await attachment.save(temp_path)
                        transcription = await voice_svc.speech_to_text(temp_path)
                        content = f"{content} [Transcribed Voice]: {transcription}"
                        if os.path.exists(temp_path): os.remove(temp_path)

        await add_chat_message(session_id, "user", content)

        async with destination.typing():
            try:
                response_text, file_path = await self.dispatcher.run(user_id, channel_id, content)
                
                if response_text:
                    await add_chat_message(session_id, "bot", response_text)
                    await destination.send(response_text)
                
                if file_path and os.path.exists(file_path):
                    filename = f"kiba_{int(time.time())}.png"
                    file = discord.File(file_path, filename=filename)
                    await destination.send(file=file)
                
                if not file_path:
                    await maybe_update_summary(self.llm, user_id, channel_id, session_id)

            except Exception as e:
                print(f"[ERROR] Dispatcher failed: {e}")
                await destination.send("❌ Neural sync error. Check terminal.")

    async def handle_natural_chat(self, message: discord.Message):
        if message.author.bot: return
        content = message.content.strip()
        if self.bot.user in message.mentions:
            mention_strings = [f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"]
            for m in mention_strings:
                content = content.replace(m, "").strip()
        if not content: return
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
        embed.add_field(name="HQ Engine", value="`FLUX.2 (Dev-4bit)`")
        embed.add_field(name="Fast Engine", value="`SDXL (Base-1.0)`")
        await ctx.send(embed=embed)

    @commands.command(aliases=["ask", "talk", "fact"])
    async def chat(self, ctx, *, message: str):
        await self.handle_chat_turn(ctx, ctx.author, ctx.channel, message)

    @commands.command(name="studio")
    async def set_studio_config(self, ctx, setting: str, value: str):
        music_service = getattr(self.bot, "music_service", None)
        if not music_service:
            return await ctx.send("❌ Music Service not loaded.")

        setting = setting.lower()
        try:
            if setting == "bpm": music_service.update_studio_settings(bpm=int(value))
            elif setting == "voice": music_service.update_studio_settings(voice=value)
            elif setting == "mode": music_service.update_studio_settings(mode=value)
            else: return await ctx.send("❓ Unknown setting. Use: bpm, voice, or mode.")
                
            embed = discord.Embed(
                title="🎼 Studio Profile Updated",
                description=f"**{setting.upper()}** set to `{value}`",
                color=discord.Color.purple()
            )
            await ctx.send(embed=embed)
        except ValueError:
            await ctx.send("❌ Invalid value.")

async def setup(bot):
    await bot.add_cog(ChatCommands(bot))