from loguru import logger
from rich.logging import RichHandler
import logging
import sys
from pathlib import Path
from hparams import Hparams

hparams = Hparams()

# setup logging file
logger.add(Path(hparams.logging_path) / "sys.log", rotation="20 MB", compression="zip")


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


logging.basicConfig(handlers=[RichHandler(markup=True)], level=20, force=True)
