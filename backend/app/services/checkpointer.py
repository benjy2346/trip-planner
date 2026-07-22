from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from app.config import get_settings

from ..logging_config import get_logger

logger = get_logger(__name__)

_saver: AsyncRedisSaver | None = None


async def init_checkpointer() -> AsyncRedisSaver:
    global _saver
    url = get_settings().redis_url
    _saver = AsyncRedisSaver(redis_url=url)
    await _saver.asetup()
    logger.info("Redis checkpointer 初始化成功")
    return _saver


async def close_checkpointer() -> None:
    global _saver
    _saver = None


def get_checkpointer() -> AsyncRedisSaver:
    if _saver is None:
        raise RuntimeError("checkpointer not initialized")
    return _saver
