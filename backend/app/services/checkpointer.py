from redis.asyncio import ConnectionPool, Redis
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from app.config import get_settings

_pool: ConnectionPool | None = None
_client: Redis | None = None
_saver: AsyncRedisSaver | None = None


async def init_checkpointer() -> AsyncRedisSaver:
    global _pool, _client, _saver
    url = get_settings().redis_url
    _pool = ConnectionPool.from_url(
        url,
        max_connections=20,
        socket_timeout=5.0,
        socket_connect_timeout=2.0,
        decode_responses=False,
        health_check_interval=30,
    )
    _client = Redis(connection_pool=_pool)
    _saver = AsyncRedisSaver(_client)
    await _saver.asetup()
    print("✅ Redis checkpointer 初始化成功")
    return _saver


async def close_checkpointer() -> None:
    global _pool, _client, _saver
    if _client:
        await _client.aclose()
    _saver = None
    _client = None
    _pool = None


def get_checkpointer() -> AsyncRedisSaver:
    if _saver is None:
        raise RuntimeError("checkpointer not initialized")
    return _saver
