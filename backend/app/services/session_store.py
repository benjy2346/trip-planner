import json
from typing import Optional
import redis.asyncio as redis
from app.config import get_settings
from app.agents.state import SupervisorState
from app.models.schemas import TripRequest, TripPlan

SESSION_TTL = 86400  # 滑动 TTL 24h
_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis_client


async def init_redis() -> None:
    client = _get_redis()
    await client.ping()
    print("✅ Redis 连接成功")


async def close_redis() -> None:
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None


def _serialize(state: SupervisorState) -> str:
    data = dict(state)
    data["trip_request"] = state["trip_request"].model_dump() if state.get("trip_request") else None
    data["trip_plan"] = state["trip_plan"].model_dump() if state.get("trip_plan") else None
    data["messages"] = [{"type": m.type, "content": m.content} for m in state.get("messages", [])]
    data["weather_outputs"] = []
    data["hotel_outputs"] = []
    data["poi_outputs"] = []
    return json.dumps(data, ensure_ascii=False)


def _deserialize(raw: str) -> SupervisorState:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    _msg_map = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}
    data = json.loads(raw)
    if data.get("trip_request"):
        data["trip_request"] = TripRequest(**data["trip_request"])
    if data.get("trip_plan"):
        data["trip_plan"] = TripPlan(**data["trip_plan"])
    data["messages"] = [
        _msg_map.get(m["type"], HumanMessage)(content=m["content"])
        for m in data.get("messages", [])
    ]
    data.setdefault("weather_outputs", [])
    data.setdefault("hotel_outputs", [])
    data.setdefault("poi_outputs", [])
    return SupervisorState(**data)


async def save_session(user_id: str, state: SupervisorState) -> None:
    client = _get_redis()
    await client.set(f"session:{user_id}", _serialize(state), ex=SESSION_TTL)


async def load_session(user_id: str) -> Optional[SupervisorState]:
    client = _get_redis()
    raw = await client.get(f"session:{user_id}")
    if raw is None:
        return None
    await client.expire(f"session:{user_id}", SESSION_TTL)
    return _deserialize(raw)
