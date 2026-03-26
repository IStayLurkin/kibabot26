# --- MUST BE AT THE ABSOLUTE TOP ---
import os
# Set HF_HOME before any huggingface imports to prevent fallback to C:/Users/.cache
os.environ.setdefault("HF_HOME", "J:/aistorage/huggingface_cache")
os.environ.setdefault("TORCH_HOME", "J:/aistorage/torch_cache")
os.environ['OLLAMA_MODELS'] = 'G:/ollamamodels'
# ----------------------------------

import asyncio
import re
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
from core.logging_config import setup_logging, get_logger, StartupProgress
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
from services.thinking_service import ThinkingService
from services.vision_service import VisionService
from services.fish_speech_service import FishSpeechService
from services.parakeet_service import ParakeetService
from services.mem0_service import Mem0Service
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
startup_progress = StartupProgress()
# Replace your current intents block with this
intents = discord.Intents.default()
intents.message_content = True
intents.members = True     # Required for the 2026 handshake
intents.presences = True   # Required to show as "Online" (Green Circle)




async def send_long_message(destination, text):
    if not text:
        return
    limit = 1900
    if len(text) <= limit:
        await destination.send(text)
        return
    # Split on sentence boundaries to avoid cutting mid-sentence
    chunks = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(current) + len(sentence) + 1 > limit:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence
    if current:
        chunks.append(current.strip())
    for chunk in chunks:
        await destination.send(chunk)


def safe_task(coro, *, name: str = ""):
    """Schedule a fire-and-forget coroutine that logs any unhandled exceptions."""
    task = asyncio.create_task(coro, name=name or None)
    def _on_done(t: asyncio.Task):
        if not t.cancelled() and t.exception() is not None:
            logger.exception("[safe_task] Background task %r raised an exception", name, exc_info=t.exception())
    task.add_done_callback(_on_done)
    return task


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
        self.search_service = None
        self.cogvideo_service = None
        self.animatediff_service = None
        self.wan_service = None
        self.thinking_service = None
        self.vision_service = None
        self.fish_speech_service = None
        self.parakeet_service = None
        self.mem0_service = None
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
        async with self.performance_tracker.track_service_call("startup.init_db"):
            await init_db()
        startup_progress.advance("Database")

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
        startup_progress.advance("Hardware")
        self.command_help_service = CommandHelpService()
        self.behavior_rule_service = BehaviorRuleService()
        self.search_service = SearchService(base_url=SEARXNG_BASE_URL, max_results=SEARXNG_MAX_RESULTS) if SEARXNG_ENABLED else None
        self.llm_service = LLMService(
            performance_tracker=self.performance_tracker,
            model_runtime_service=self.model_runtime_service,
            behavior_rule_service=self.behavior_rule_service,
            search_service=self.search_service,
        )
        from database.behavior_rules_repository import get_bot_config
        from services.llm_service import PERSONALITIES, DEFAULT_PERSONALITY
        saved_personality = await get_bot_config("active_personality", DEFAULT_PERSONALITY)
        if saved_personality in PERSONALITIES:
            self.llm_service.active_personality = saved_personality
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
        self.thinking_service = ThinkingService(performance_tracker=self.performance_tracker)
        self.vision_service = VisionService(performance_tracker=self.performance_tracker)
        from core.config import FISH_SPEECH_ENABLED, PARAKEET_ENABLED
        if FISH_SPEECH_ENABLED:
            self.fish_speech_service = FishSpeechService(performance_tracker=self.performance_tracker)
        if PARAKEET_ENABLED:
            self.parakeet_service = ParakeetService(performance_tracker=self.performance_tracker)
        from core.config import MEM0_ENABLED
        if MEM0_ENABLED:
            try:
                self.mem0_service = Mem0Service()
            except Exception as exc:
                logger.warning("[startup] Mem0 failed to initialize: %s", exc)
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
        startup_progress.advance("Services")
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
            "cogs.thinking_commands",
            "cogs.vision_commands",
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

        startup_progress.advance("Cogs")
        self.task_manager.start_all()

    async def close(self):
        logger.info("Shutting down background tasks...")
        self.task_manager.stop_all()
        try:
            from database.db_connection import close_db
            await close_db()
        except Exception as exc:
            logger.warning("Error closing DB connection: %s", exc)
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
    await bot.change_presence(
        status=discord.Status.online,
        activity=discord.Activity(type=discord.ActivityType.listening, name="!help"),
    )

    if not bot.startup_banner_printed:
        startup_progress.advance("Discord")
        bot.print_startup_banner()
        bot.startup_banner_printed = True
        safe_task(_prewarm_ollama(), name="prewarm_ollama")
        safe_task(_validate_services(bot), name="validate_services")
    else:
        logger.info("[gateway] reconnected user=%s", bot.user)
        await bot.change_presence(
            status=discord.Status.online,
            activity=discord.Activity(type=discord.ActivityType.listening, name="!help"),
        )


async def _prewarm_ollama():
    """Send a minimal dummy request so Ollama loads the model into VRAM before the first real message."""
    if bot.llm_service is None:
        return
    try:
        await bot.llm_service.generate_text(".")
        logger.info("[prewarm] Ollama model loaded into VRAM and ready.")
    except Exception as exc:
        logger.warning("[prewarm] Ollama pre-warm failed (will load on first message): %s", exc)
    finally:
        startup_progress.advance("Ollama")


async def _validate_services(b: "ExpenseBot") -> None:
    """Log warnings for optional services that are misconfigured or unavailable."""
    from core.config import GIPHY_API_KEY, SEARXNG_ENABLED, SEARXNG_BASE_URL

    if not GIPHY_API_KEY:
        logger.warning("[startup] GIPHY_API_KEY not set — image search will only use local folder")

    if SEARXNG_ENABLED:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{SEARXNG_BASE_URL}/healthz")
            if resp.status_code != 200:
                logger.warning("[startup] SearXNG returned %d — web search may not work", resp.status_code)
            else:
                logger.info("[startup] SearXNG reachable at %s", SEARXNG_BASE_URL)
        except Exception as exc:
            logger.warning("[startup] SearXNG not reachable: %s — web search may not work", exc)
        finally:
            startup_progress.advance("SearXNG")

    if b.vector_memory_service is not None:
        try:
            test_vec = await b.vector_memory_service._embed.embed("test")
            if not test_vec:
                logger.warning("[startup] Embedding service returned empty — vector memory will not work")
            else:
                logger.info("[startup] Embedding service OK (%d dims)", len(test_vec))
        except Exception as exc:
            logger.warning("[startup] Embedding service failed: %s", exc)
        finally:
            startup_progress.advance("Embeddings")


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
    loop = asyncio.get_running_loop()
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
