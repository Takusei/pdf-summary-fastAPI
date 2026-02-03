from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from app.cache.utils import VDR_DB_DIR
from app.core.config import settings

_LOGGERS: dict[str, logging.Logger] = {}
_LOG_BASE_DIR: ContextVar[str | None] = ContextVar("log_base_dir", default=None)


def set_log_base_dir(base_dir: str | Path | None) -> None:
    if base_dir is None:
        _LOG_BASE_DIR.set(None)
        return
    _LOG_BASE_DIR.set(str(Path(base_dir)))


@contextmanager
def log_base_dir(base_dir: str | Path | None) -> Iterator[None]:
    token = _LOG_BASE_DIR.set(str(Path(base_dir)) if base_dir else None)
    try:
        yield
    finally:
        _LOG_BASE_DIR.reset(token)


def _resolve_log_path() -> Path:
    base_dir = _LOG_BASE_DIR.get()
    if base_dir:
        log_name = Path(settings.LOG_FILE_PATH).name
        log_path = Path(base_dir) / VDR_DB_DIR / log_name
    else:
        log_path = Path(settings.LOG_FILE_PATH)
        if not log_path.is_absolute():
            log_path = Path(os.getcwd()) / log_path
    return log_path


def _ensure_logger() -> logging.Logger:
    log_path = _resolve_log_path()
    key = str(log_path)
    if key in _LOGGERS:
        return _LOGGERS[key]

    logger = logging.getLogger(f"app.timing.{abs(hash(key))}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    _LOGGERS[key] = logger
    return logger


def log_event(event: str, duration_s: float | None = None, **fields: Any) -> None:
    logger = _ensure_logger()
    parts = [f"event={event}"]
    if duration_s is not None:
        parts.append(f"duration_s={duration_s:.4f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))
