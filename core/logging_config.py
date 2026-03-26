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

    def __init__(self):
        global _startup_active
        self._current = 0
        self._total = len(self.STEPS)
        _startup_active = True

    def advance(self, label: str | None = None):
        global _startup_active
        self._current = min(self._current + 1, self._total)
        self._print(label)
        if self._current >= self._total:
            _startup_active = False
            sys.stdout.write("\n")
            sys.stdout.flush()
            if _real_handler is not None:
                for record in _log_buffer:
                    _real_handler.emit(record)
                _log_buffer.clear()

    def _print(self, label: str | None = None):
        filled = int(self.BAR_WIDTH * self._current / self._total)
        empty = self.BAR_WIDTH - filled
        bar = f"{self.GREEN}{'█' * filled}{self.GREY}{'░' * empty}{RESET}"
        pct = int(100 * self._current / self._total)
        if self._current >= self._total:
            status = f"{self.GREEN}Kiba Bot ready!{RESET}"
        else:
            status = f"{self.CYAN}Kiba Bot loading...{RESET}"
        line = f"\r[{bar}] {pct:3d}%  {status}"
        sys.stdout.write(line + "\033[K")
        sys.stdout.flush()
