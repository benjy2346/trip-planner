"""统一日志配置：级别可配、带请求 ID、可被日志采集系统解析。

用法：
    from app.logging_config import get_logger, setup_logging
    setup_logging()                    # 进程启动时调用一次
    logger = get_logger(__name__)      # 每个模块各取各的
    logger.info("...")

请求 ID 通过 contextvar 传递，同一请求在各处打出的日志共享同一个 ID，
便于在日志里把一次请求的完整链路串起来。
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def new_request_id() -> str:
    """生成短请求 ID（8 位足够在单机日志里区分）。"""
    return uuid.uuid4().hex[:8]


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    """把当前请求 ID 注入日志记录，让 formatter 能引用 %(request_id)s。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def setup_logging(level: str | None = None) -> None:
    """配置根 logger。重复调用是安全的（会先清掉自己装过的 handler）。"""
    if level is None:
        from app.config import settings

        level = settings.log_level

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_trip_planner_handler", False):
            root.removeHandler(existing)
    handler._trip_planner_handler = True  # type: ignore[attr-defined]

    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """取模块级 logger；未配置过则保证 filter 存在，避免 formatter 报错。"""
    logger = logging.getLogger(name)
    if not any(isinstance(f, _RequestIdFilter) for f in logger.filters):
        logger.addFilter(_RequestIdFilter())
    return logger
