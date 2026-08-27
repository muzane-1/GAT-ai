"""Centralised logging helpers.

Every module in the system requests its logger through :func:`get_logger`
so handler configuration happens exactly once and notebooks/scripts get a
consistent, timestamped format.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_INITIALIZED: set[str] = set()


def get_logger(name: str, log_file: Path | str | None = None) -> logging.Logger:
    """Return an idempotent, consistently formatted logger.

    Args:
        name: Logger name, typically ``__name__`` of the caller.
        log_file: Optional file path. When provided a rotating file handler
            is attached in addition to stdout.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if name in _INITIALIZED:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=2)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)

    logger.propagate = False
    _INITIALIZED.add(name)
    return logger
