import logging
import os
import sys
import warnings


RESET = "\033[0m"
DISCORD_PURPLE = "\033[38;5;177m"
WARNING_YELLOW = "\033[38;5;226m"
ERROR_RED = "\033[38;5;196m"


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


_startup_active = False
_log_buffer: list = []


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)

        if record.levelno >= logging.ERROR:
            return f"{ERROR_RED}{rendered}{RESET}"

        if record.levelno >= logging.WARNING:
            return f"{WARNING_YELLOW}{rendered}{RESET}"

        return f"{DISCORD_PURPLE}{rendered}{RESET}"


class BufferingHandler(logging.Handler):
    """Buffers log records during startup, flushes to real handler after."""
    def __init__(self, real_handler: logging.Handler):
        super().__init__()
        self.real_handler = real_handler

    def emit(self, record: logging.LogRecord):
        if _startup_active:
            _log_buffer.append(record)
        else:
            self.real_handler.emit(record)


def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        if handle == 0:
            return

        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return

        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        return


def setup_logging(level: int = logging.INFO):
    global _real_handler
    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*", category=UserWarning)
    _enable_windows_ansi()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    _real_handler = logging.StreamHandler()
    _real_handler.setFormatter(ColorFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(BufferingHandler(_real_handler))

    # Always write plain logs to bot.log regardless of terminal piping
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    plain_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(plain_formatter)
    root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("discord.client").setLevel(logging.WARNING)
    logging.getLogger("discord.gateway").setLevel(logging.WARNING)


_real_handler: logging.Handler | None = None


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class StartupProgress:
    STEPS = [
        "Database",
        "Hardware",
        "Services",
        "Cogs",
        "Discord",
        "SearXNG",
        "Embeddings",
        "Ollama",
    ]
    BAR_WIDTH = 30
    GREEN = "\033[38;5;82m"
    CYAN = "\033[38;5;51m"
    GREY = "\033[38;5;240m"
    OWNER_ID = 1401400377110171808

    def __init__(self):
        global _startup_active
        self._current = 0
        self._total = len(self.STEPS)
        self._done = False
        _startup_active = True
        self._discord_message = None  # set after bot is ready
        self._bot = None

    async def attach_bot(self, bot):
        """Call this once the Discord client is ready to enable DM progress updates."""
        self._bot = bot
        await self._load_restart_state()

    async def _load_restart_state(self):
        import json
        from pathlib import Path
        state_file = Path(__file__).parent.parent / ".restart_state.json"
        if not state_file.exists():
            return
        try:
            state = json.loads(state_file.read_text())
            channel = await self._bot.fetch_channel(state["channel_id"])
            self._discord_message = await channel.fetch_message(state["message_id"])
            state_file.unlink()
        except Exception:
            pass

    def advance(self, label: str | None = None):
        global _startup_active
        if self._done:
            return
        self._current = min(self._current + 1, self._total)
        self._print(label)
        if self._bot is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._update_discord())
            except Exception:
                pass
        if self._current >= self._total:
            self._done = True
            _startup_active = False
            sys.stdout.write("\n")
            sys.stdout.flush()
            if _real_handler is not None:
                for record in _log_buffer:
                    _real_handler.emit(record)
                _log_buffer.clear()

    def _bar_text(self) -> str:
        filled = int(self.BAR_WIDTH * self._current / self._total)
        empty = self.BAR_WIDTH - filled
        bar = "█" * filled + "░" * empty
        pct = int(100 * self._current / self._total)
        if self._current >= self._total:
            status = "Koba ready!"
        else:
            step = self.STEPS[min(self._current, self._total - 1)]
            status = f"Loading {step}..."
        return f"[{bar}] {pct}%  {status}"

    async def _update_discord(self):
        import asyncio
        try:
            text = f"```\n{self._bar_text()}\n```"
            if self._discord_message is not None:
                await self._discord_message.edit(content=text)
                await asyncio.sleep(0.3)
            else:
                user = await self._bot.fetch_user(self.OWNER_ID)
                dm = await user.create_dm()
                self._discord_message = await dm.send(text)
        except Exception:
            pass

    def _print(self, label: str | None = None):
        filled = int(self.BAR_WIDTH * self._current / self._total)
        empty = self.BAR_WIDTH - filled
        bar = f"{self.GREEN}{'█' * filled}{self.GREY}{'░' * empty}{RESET}"
        pct = int(100 * self._current / self._total)
        if self._current >= self._total:
            status = f"{self.GREEN}Kiba Bot ready!{RESET}"
        else:
            status = f"{self.CYAN}Kiba Bot loading...{RESET}"
        sys.stdout.write(f"\r[{bar}] {pct:3d}%  {status}\033[K")
        sys.stdout.flush()
