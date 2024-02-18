from loguru import logger
from rich.logging import RichHandler
from rich.console import Console
import logging
import sys
from pathlib import Path
from hparams import Hparams

hparams = Hparams()
console = Console()
logger.remove()


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _log_formatter(record: dict) -> str:
    """Log message formatter"""
    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "cyan",
        "INFO": "bold",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }
    lvl_color = color_map.get(record["level"].name, "cyan")
    return (
        "[not bold green]{time:YYYY/MM/DD HH:mm:ss}[/not bold green] | {level.icon}"
        + f"  - [{lvl_color}]{{message}}[/{lvl_color}]"
    )


# setup logging file
logger.add(Path(hparams.logging_path) / "sys.log", rotation="20 MB", compression="zip")
# setup rich passthrough
logger.add(
    console.print,
    level="TRACE",
    format=_log_formatter,
    colorize=True,
)

logging.basicConfig(handlers=[RichHandler()], level=20, force=True)
