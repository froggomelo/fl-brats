"""Application logging setup."""

import logging
from pathlib import Path
from typing import Any

from fl_brats.utils.paths import ensure_directory


def configure_logging(config: Any) -> logging.Logger:
    """Configure non-duplicated console and optional per-run file logging."""
    logger = logging.getLogger("fl_brats")
    logger.setLevel(getattr(logging, config.level))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if config.console:
        handler: logging.Handler
        if config.rich:
            try:
                from rich.logging import RichHandler

                handler = RichHandler(rich_tracebacks=True)
            except ImportError:
                handler = logging.StreamHandler()
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if config.file:
        filename = Path(config.filename)
        ensure_directory(filename.parent)
        file_handler = logging.FileHandler(filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
