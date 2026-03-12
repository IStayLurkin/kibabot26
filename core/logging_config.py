import logging
import os


RESET = "\033[0m"
DISCORD_PURPLE = "\033[38;5;177m"
HTB_GREEN = "\033[38;5;82m"
BRIGHT_GREEN = "\033[38;5;118m"
WARNING_YELLOW = "\033[38;5;226m"
ERROR_RED = "\033[38;5;196m"


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        message = record.getMessage()

        if message.startswith("Bot starting in"):
            return f"{DISCORD_PURPLE}{rendered}{RESET}"

        if record.levelno >= logging.ERROR:
            return f"{ERROR_RED}{rendered}{RESET}"

        if record.levelno >= logging.WARNING:
            return f"{WARNING_YELLOW}{rendered}{RESET}"

        success_prefixes = (
            "Bot started.",
            "Ping:",
            "Switched ",
            "Model ready:",
            "Pulling AI model:",
            "Reply sent",
            "Generating ",
        )
        if message.startswith(success_prefixes):
            return f"{BRIGHT_GREEN}{rendered}{RESET}"

        return f"{HTB_GREEN}{rendered}{RESET}"


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
    _enable_windows_ansi()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("discord.client").setLevel(logging.WARNING)
    logging.getLogger("discord.gateway").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
