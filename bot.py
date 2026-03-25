# --- MUST BE AT THE ABSOLUTE TOP ---
import os
# Force all 33GB+ of AI models to stay on your G: drive project folder

# Redirection for Ollama (Qwen3) - Ensures the bot knows where the engine sits
os.environ['OLLAMA_MODELS'] = 'G:/ollamamodels'
# ----------------------------------
# ----------------------------------

import asyncio
import subprocess
import sys

import time
import discord
from discord.ext import commands
import ctypes.util
from core.config import (
    DISCORD_BOT_TOKEN,
    BOT_PREFIX,
    BOT_TIMEZONE,
    OLLAMA_BASE_URL,
    SEARXNG_ENABLED,
    SEARXNG_BASE_URL,
    SEARXNG_MAX_RESULTS,
)
from core.logging_config import setup_logging, get_logger
from database.database import init_db
from services.behavior_rule_service import BehaviorRuleService
from services.code_execution_service import CodeExecutionService
from services.codegen_service import CodegenService
from services.command_help_service import CommandHelpService
from services.hardware_service import HardwareService
from services.image_service import ImageService
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.model_storage_service import ModelStorageService
from services.vector_memory_service import VectorMemoryService
from services.search_service import SearchService
from services.music_service import MusicService
from services.model_runtime_service import ModelRuntimeService
from services.osint_service import OSINTService
from services.performance_service import PerformanceTracker
from services.song_session_service import SongSessionService
from services.video_service import VideoService
from services.voice_service import VoiceService
from services.cogvideo_service import CogVideoService
from services.animatediff_service import AnimateDiffService
from services.wan_service import WanService
from tasks.task_manager import TaskManager


# 2026 DAVE/Opus Hardware Bridge
if not discord.opus.is_loaded():
    opus_path = ctypes.util.find_library('opus')
    if opus_path:
        discord.opus.load_opus(opus_path)
    else:
        # Fallback for hardened Kali paths
        pass  # Linux path skipped on Windows

setup_logging()
logger = get_logger(__name__)
# Replace your current intents block with this
intents = discord.Intents.default()
intents.message_content = True
intents.members = True     # Required for the 2026 handshake
intents.presences = True   # Required to show as "Online" (Green Circle)


async def _ollama_is_running() -> bool:
    """Return True if ollama is already responding on its API port."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", 11434), timeout=1.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def _wait_for_ollama(timeout: float = 30.0) -> bool:
    """Poll until ollama is ready or timeout expires. Returns True if ready."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await _ollama_is_running():
            return True
        await asyncio.sleep(1.0)
    return False


async def send_long_message(destination, text):
    if not text:
        return
    # Splits by 1900 to stay under 2000 limit
    for i in range(0, len(text), 1900):
        await destination.send(text[i:i+1900])

class ExpenseBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_banner_printed = False
        self.performance_tracker = PerformanceTracker()
        self.task_manager = TaskManager(self)
        self.song_session_service = SongSessionService()
        
        # Hardware-Aware State Flags
        # Counter + lock prevent the VRAM Guard from clearing memory during active generation.
        # Use bot.generating_lock with async with, increment generating_count on start,
        # decrement on finish. VRAMGuard checks generating_count > 0.
        self.generating_count = 0
        self.generating_lock = asyncio.Lock()
        
        self._ollama_process = None
        self.llm_service = None
        self.image_service = None
        self.voice_service = None
        self.video_service = None
        self.music_service = None
        self.codegen_service = None
        self.code_execution_service = None
        self.behavior_rule_service = None
        self.osint_service = None
        self.hardware_service = None
        self.model_storage_service = None
        self.model_runtime_service = None
        self.command_help_service = None
        self.vector_memory_service = None
        self.cogvideo_service = None
        self.animatediff_service = None
        self.wan_service = None
        self.start_time = time.perf_counter()

    async def on_message(self, message):
            # 1. Ignore yourself and other bots (Stability check)
            if message.author.bot:
                return

            # 2. MANDATORY: The "Command Pass-Through"
            # This tells the bot to look for prefixes like '!' and run the associated code.
            await self.process_commands(message)

            # 3. Handle Natural Chat (AI logic)
            # We only run this if the message WASN'T a valid command.
            ctx = await self.get_context(message)
            if not ctx.valid:
                if self.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
                    chat_cog = self.get_cog("ChatCommands")
                    if chat_cog:
                        # Routes to your Qwen3/LLM bridge
                        await chat_cog.handle_natural_chat(message)
           

    async def setup_hook(self):
        # Start ollama serve if not already running
        if not await _ollama_is_running():
            logger.info("[ollama] Not running — launching ollama serve...")
            self._ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            ready = await _wait_for_ollama(timeout=30.0)
            if ready:
                logger.info("[ollama] Ready (pid %d)", self._ollama_process.pid)
            else:
                logger.warning("[ollama] Did not become ready within 30s — LLM calls may fail")
        else:
            logger.info("[ollama] Already running — skipping launch")

        async with self.performance_tracker.track_service_call("startup.init_db"):
            await init_db()

        service_started_at = time.perf_counter()
        self.hardware_service = HardwareService()
        self.model_storage_service = ModelStorageService(
            performance_tracker=self.performance_tracker,
        )
        self.model_storage_service.initialize_storage()
        self.model_runtime_service = ModelRuntimeService(
            hardware_service=self.hardware_service,
            model_storage_service=self.model_storage_service,
            performance_tracker=self.performance_tracker,
        )
        await self.model_runtime_service.initialize()
        self.command_help_service = CommandHelpService()
        self.behavior_rule_service = BehaviorRuleService()
        search_service = SearchService(base_url=SEARXNG_BASE_URL, max_results=SEARXNG_MAX_RESULTS) if SEARXNG_ENABLED else None
        self.llm_service = LLMService(
            performance_tracker=self.performance_tracker,
            model_runtime_service=self.model_runtime_service,
            behavior_rule_service=self.behavior_rule_service,
            search_service=search_service,
        )
        _embed_base = OLLAMA_BASE_URL[:-3] if OLLAMA_BASE_URL.endswith("/v1") else OLLAMA_BASE_URL
        embedding_service = EmbeddingService(base_url=_embed_base, model="nomic-embed-text")
        self.vector_memory_service = VectorMemoryService(embedding_service=embedding_service, top_k=5)
        self.image_service = ImageService(
            performance_tracker=self.performance_tracker,
        )
        self.voice_service = VoiceService(
            llm_service=self.llm_service,
            performance_tracker=self.performance_tracker,
        )
        self.video_service = VideoService(
            llm_service=self.llm_service,
            performance_tracker=self.performance_tracker,
        )
        self.cogvideo_service = CogVideoService()
        self.animatediff_service = AnimateDiffService()
        self.wan_service = WanService(runtime_service=self.model_runtime_service)
        self.music_service = MusicService(
            performance_tracker=self.performance_tracker,
            runtime_service=self.model_runtime_service,
        )
        self.codegen_service = CodegenService(
            llm_service=self.llm_service,
            performance_tracker=self.performance_tracker,
        )
        self.code_execution_service = CodeExecutionService(
            performance_tracker=self.performance_tracker,
        )
        self.code_execution_service.initialize_workspace()
        self.osint_service = OSINTService(performance_tracker=self.performance_tracker)
        self.performance_tracker.record_service_call(
            "startup.init_services",
            (time.perf_counter() - service_started_at) * 1000,
        )

        extensions = [
            "cogs.expense_commands",
            "cogs.budget_commands",
            "cogs.error_handler",
            "cogs.chat_commands",
            "cogs.dev_commands",
            "cogs.media_commands",
            "cogs.agent_commands",
            "cogs.runtime_commands",
            "cogs.code_commands",
            "cogs.video_commands",
            "tasks.vram_guard", # Essential VRAM monitoring task
        ]

        for extension in extensions:
            logger.debug("Loading extension %s", extension)
            started_at = time.perf_counter()
            try:
                await self.load_extension(extension)
                self.performance_tracker.record_service_call(
                    f"startup.load_extension.{extension}",
                    (time.perf_counter() - started_at) * 1000,
                )
            except Exception as e:
                logger.error("Failed to load extension %s: %s", extension, e)

        self.task_manager.start_all()

    async def close(self):
        logger.info("Shutting down background tasks...")
        self.task_manager.stop_all()
        try:
            from database.db_connection import close_db
            await close_db()
        except Exception as exc:
            logger.warning("Error closing DB connection: %s", exc)
        if self._ollama_process is not None:
            logger.info("[ollama] Terminating ollama serve (pid %d)...", self._ollama_process.pid)
            self._ollama_process.terminate()
            try:
                self._ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ollama_process.kill()
        await super().close()

    def print_startup_banner(self):
        chat_cog = self.get_cog("ChatCommands")
        provider = "unknown"
        model = "unknown"

        if chat_cog is not None and getattr(chat_cog, "llm", None) is not None:
            provider = chat_cog.llm.provider
            model = chat_cog.llm._get_active_model_name()
        elif self.llm_service is not None:
            provider = getattr(self.llm_service, "provider", "unknown")
            get_model_name = getattr(self.llm_service, "_get_active_model_name", None)
            if callable(get_model_name):
                model = get_model_name()

        loaded_cogs = sorted(self.extensions.keys())
        startup_snapshot = self.performance_tracker.get_health_snapshot()
        startup_duration_ms = (time.perf_counter() - self.start_time) * 1000

        logger.info("Bot started.")
        logger.info("Ping: %sms", f"{self.latency * 1000:.0f}")
        logger.debug(
            "[startup] user=%s prefix=%s provider=%s model=%s timezone=%s cogs=%s startup_ms=%.2f ws=%sms services=%s commands=%s",
            self.user,
            BOT_PREFIX,
            provider,
            model,
            BOT_TIMEZONE,
            len(loaded_cogs),
            startup_duration_ms,
            f"{startup_snapshot['websocket_current_ms']:.0f}",
            len(startup_snapshot["services"]),
            len(startup_snapshot["commands"]),
        )
        if self.model_runtime_service is not None:
            logger.debug("[startup] runtime_state=%s", self.model_runtime_service.get_runtime_snapshot())
        logger.debug("Loaded cogs: %s", ", ".join(loaded_cogs))


bot = ExpenseBot(command_prefix=BOT_PREFIX, intents=intents, help_command=None)


@bot.event
async def on_ready():
    # Explicitly set status to bypass sidebar sync lag
    await bot.change_presence(status=discord.Status.online)

    if not bot.startup_banner_printed:
        bot.print_startup_banner()
        bot.startup_banner_printed = True
        asyncio.create_task(_prewarm_ollama())
    else:
        logger.info("[gateway] reconnected user=%s", bot.user)


async def _prewarm_ollama():
    """Send a minimal dummy request so Ollama loads the model into VRAM before the first real message."""
    if bot.llm_service is None:
        return
    try:
        await bot.llm_service.generate_text(".")
        logger.info("[prewarm] Ollama model loaded into VRAM and ready.")
    except Exception as exc:
        logger.warning("[prewarm] Ollama pre-warm failed (will load on first message): %s", exc)

@bot.event
async def on_resumed():
    logger.info("[gateway] session resumed")


@bot.event
async def on_disconnect():
    logger.warning("[gateway] disconnected")


@bot.before_invoke
async def before_any_command(ctx: commands.Context):
    if ctx.command is None:
        return

    bot.performance_tracker.start_command(id(ctx.message), ctx.command.qualified_name)


@bot.after_invoke
async def after_any_command(ctx: commands.Context):
    if ctx.command is None:
        return

    duration_ms = bot.performance_tracker.finish_command(id(ctx.message))
    if duration_ms is None:
        return

    if duration_ms >= 1000:
        logger.info(
            "[command] name=%s duration_ms=%.2f websocket_latency_ms=%.2f",
            ctx.command.qualified_name,
            duration_ms,
            bot.latency * 1000,
        )
    else:
        logger.debug(
            "[command] name=%s duration_ms=%.2f websocket_latency_ms=%.2f",
            ctx.command.qualified_name,
            duration_ms,
            bot.latency * 1000,
        )


async def main():
    if not DISCORD_BOT_TOKEN:
        raise ValueError("Error: DISCORD_BOT_TOKEN not found in environment variables.")

    # Increase default thread pool so asyncio.to_thread LLM calls
    # don't starve the Discord heartbeat on long inference.
    import concurrent.futures
    loop = asyncio.get_event_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=8))

    logger.debug("Bot initializing...")

    async with bot:
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                await bot.start(DISCORD_BOT_TOKEN)
                return
            except discord.DiscordServerError as exc:
                if attempt >= max_attempts:
                    raise
                retry_delay = min(30, attempt * 5)
                logger.warning(
                    "Discord is temporarily unavailable. Retrying in %ss (%s/%s).",
                    retry_delay,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(retry_delay)
            except discord.HTTPException as exc:
                if getattr(exc, "status", 0) < 500 or attempt >= max_attempts:
                    raise
                retry_delay = min(30, attempt * 5)
                logger.warning(
                    "Discord returned a server error. Retrying in %ss (%s/%s).",
                    retry_delay,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(retry_delay)


if __name__ == "__main__":
    asyncio.run(main())
