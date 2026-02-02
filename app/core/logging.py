from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from app.core.config import settings

_LOGGER: logging.Logger | None = None


def _ensure_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("app.timing")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path = Path(settings.LOG_FILE_PATH)
        if not log_path.is_absolute():
            log_path = Path(os.getcwd()) / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    _LOGGER = logger
    return logger


def log_event(event: str, duration_s: float | None = None, **fields: Any) -> None:
    logger = _ensure_logger()
    parts = [f"event={event}"]
    if duration_s is not None:
        parts.append(f"duration_s={duration_s:.4f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))
