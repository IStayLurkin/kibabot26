import os
import asyncio
import time
import gc
import torch
import discord
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bot import send_long_message
from discord.ext import commands

from core.constants import (
    BOT_ALLOWED_CHAT_CHANNELS,
    CHAT_COOLDOWN_SECONDS,
)
from core.config import GALLERY_CHANNEL_ID, GPU_TOTAL_VRAM_MB
from database.chat_memory import (
    add_chat_message,
    get_or_create_session,
)
from services.llm_service import LLMService
from services.agent_dispatcher import AgentDispatcher
from services.chat_service import generate_dynamic_reply
from services.summary_service import maybe_update_summary
from core.logging_config import get_logger

logger = get_logger(__name__)

class ChatCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.user_cooldowns = {}
        self.llm = getattr(bot, "llm_service", None) or LLMService()
        self.allowed_chat_channels = BOT_ALLOWED_CHAT_CHANNELS
        self.dispatcher = AgentDispatcher(bot)
        self.image_service = getattr(bot, "image_service", None)
        self.hardware_service = getattr(bot, "hardware_service", None)

    def _build_services(self) -> dict:
        """Assemble the services dict expected by chat_service.generate_dynamic_reply."""
        bot = self.bot
        return {
            "llm": self.llm,
            "bot": bot,
            "image_service": getattr(bot, "image_service", None),
            "voice_service": getattr(bot, "voice_service", None),
            "video_service": getattr(bot, "video_service", None),
            "music_service": getattr(bot, "music_service", None),
            "osint_service": getattr(bot, "osint_service", None),
            "codegen_service": getattr(bot, "codegen_service", None),
            "model_runtime_service": getattr(bot, "model_runtime_service", None),
            "command_help_service": getattr(bot, "command_help_service", None),
            "behavior_rule_service": getattr(bot, "behavior_rule_service", None),
        }

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
        used_vram = self.hardware_service.get_vram_usage_mb() if self.hardware_service else 0
        total_vram = GPU_TOTAL_VRAM_MB
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

    def _get_vram_usage(self) -> int:
        if self.hardware_service:
            return self.hardware_service.get_vram_usage_mb()
        return 0

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
    @commands.cooldown(1, 60, commands.BucketType.user)
    async def dossier(self, ctx, target: str):
        """Triggers the new 2026 Agentic OSINT Research Loop."""
        osint_svc = getattr(self.bot, "osint_service", None)
        if not osint_svc:
            return await ctx.send("❌ OSINT Service not active.")

        async with ctx.typing():
            report = await osint_svc.run_dossier(target)
            await send_long_message(ctx, report)

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

        # Enhance prompt via Ollama
        enhanced_prompt = await self.llm.enhance_image_prompt(prompt)
        if enhanced_prompt != prompt:
            await status_msg.edit(content=f"{icon} **Prompt enhanced. Rendering ({mode})...**\n[░░░░░░░░░░] 0%")

        # Call the unified service logic
        if mode == "SDXL":
            path = await self.image_service.generate_sdxl(enhanced_prompt, progress_callback=update_bar)
        else:
            path = await self.image_service.generate_image(enhanced_prompt, progress_callback=update_bar)

        if path:
            await status_msg.edit(content=f"✅ **{mode} Generation Complete!**")

            image_file = discord.File(path, filename=f"kiba_{mode.lower()}.png")
            await ctx.send(content=f"Request: *{enhanced_prompt}* ({mode} Engine)", file=image_file)

            # Archive to Gallery
            gallery_channel = self.bot.get_channel(int(GALLERY_CHANNEL_ID)) if GALLERY_CHANNEL_ID else None
            if gallery_channel:
                archive_file = discord.File(path, filename=f"archive_{mode.lower()}.png")
                await gallery_channel.send(
                    content=f"🖼️ **Engine:** {mode}\n👤 **User:** {ctx.author.mention}\n📝 **Prompt:** {enhanced_prompt}",
                    file=archive_file
                )
        else:
            await status_msg.edit(content=f"❌ **{mode} Engine Error.** Check terminal.")

    # --- CORE CHAT LOGIC ---

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
                        try:
                            transcription = await voice_svc.speech_to_text(temp_path)
                            content = f"{content} [Transcribed Voice]: {transcription}"
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

        await add_chat_message(session_id, "user", content)

        async with destination.typing():
            try:
                intent = self.dispatcher.classify_intent(content)

                if intent in ("draw", "sing"):
                    # Media path — AgentDispatcher handles VRAM locking and generation
                    response_text, file_path = await self.dispatcher.run(user_id, channel_id, content)

                    if response_text:
                        await add_chat_message(session_id, "bot", response_text)
                        await send_long_message(destination, response_text)

                    if file_path and os.path.exists(file_path):
                        filename = f"kiba_{int(time.time())}.png"
                        await destination.send(file=discord.File(file_path, filename=filename))

                else:
                    # Text path — full chat_service pipeline with tool routing,
                    # agentic planning, behavior rules, and graceful fallback chains
                    display_name = getattr(author, "display_name", str(author))
                    reply = await generate_dynamic_reply(
                        llm=self.llm,
                        display_name=display_name,
                        user_id=user_id,
                        channel_id=channel_id,
                        session_id=session_id,
                        user_text=content,
                        services=self._build_services(),
                    )

                    if reply.content:
                        await add_chat_message(session_id, "bot", reply.content)
                        content = reply.content
                        if len(content) <= 200:
                            await destination.send(content)
                        else:
                            # Chunked streaming-style delivery
                            chunk_size = 250
                            placeholder = await destination.send(content[:chunk_size])
                            for i in range(chunk_size, len(content), chunk_size):
                                await asyncio.sleep(0.05)
                                end = min(i + chunk_size, len(content))
                                await placeholder.edit(content=content[:end])
                            # If content > 2000, send remaining in follow-up messages
                            if len(content) > 2000:
                                for i in range(2000, len(content), 1900):
                                    await destination.send(content[i:i+1900])

                    for fp in reply.file_paths:
                        if fp and os.path.exists(fp):
                            filename = f"kiba_{int(time.time())}.png"
                            await destination.send(file=discord.File(fp, filename=filename))

            except Exception:
                logger.exception("Chat turn failed")
                await destination.send("❌ Neural sync error. Check terminal.")

        # Run summary update after typing indicator is closed — silent background task
        asyncio.create_task(maybe_update_summary(self.llm, user_id, channel_id, session_id))

    async def handle_natural_chat(self, message: discord.Message):
        if message.author.bot:
            return
        if not self.is_allowed_chat_channel(message):
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
        embed.add_field(name="HQ Engine", value="`FLUX.2 (Dev-4bit)`")
        embed.add_field(name="Fast Engine", value="`SDXL (Base-1.0)`")
        await ctx.send(embed=embed)

    @commands.command(name="forget")
    async def forget_history(self, ctx):
        """Clears your chat history and memory with Kiba in this channel."""
        from database.chat_memory import delete_user_history
        user_id = str(ctx.author.id)
        channel_id = str(ctx.channel.id)
        await delete_user_history(user_id, channel_id)
        embed = discord.Embed(
            title="🧹 Memory Cleared",
            description="Your chat history and memory in this channel have been wiped. Starting fresh.",
            color=discord.Color.red(),
        )
        await ctx.send(embed=embed)

    @commands.command(name="models")
    async def loaded_models(self, ctx):
        """Shows which AI models are currently loaded in VRAM via Ollama."""
        models = await asyncio.to_thread(self.hardware_service.get_ollama_running_models)
        embed = discord.Embed(title="🤖 Active AI Models", color=discord.Color.dark_blue())
        if not models:
            embed.description = "No models currently loaded in VRAM."
        else:
            for m in models:
                name = m.get("name", "unknown")
                size_gb = round(m.get("size", 0) / 1e9, 1)
                vram_size = m.get("size_vram", 0)
                vram_gb = round(vram_size / 1e9, 1) if vram_size else "?"
                embed.add_field(name=name, value=f"Size: {size_gb}GB | VRAM: {vram_gb}GB", inline=False)
        await ctx.send(embed=embed)

    @commands.command(name="purge")
    @commands.is_owner()
    async def purge_channel(self, ctx):
        """[Owner only] Wipes all chat history for every user in this channel."""
        from database.chat_memory import delete_channel_history
        channel_id = str(ctx.channel.id)
        await delete_channel_history(channel_id)
        embed = discord.Embed(
            title="🗑️ Channel History Purged",
            description=f"All chat history for **#{ctx.channel.name}** has been deleted.",
            color=discord.Color.dark_red(),
        )
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
