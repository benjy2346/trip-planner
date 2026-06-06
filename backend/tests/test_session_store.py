import pytest
import fakeredis.aioredis
from unittest.mock import patch
from app.models.schemas import TripRequest
from app.agents.state import SupervisorState


def _make_state() -> SupervisorState:
    return SupervisorState(
        trip_request=TripRequest(
            user_id="u1", city="北京", start_date="2025-06-01", end_date="2025-06-03",
            travel_days=3, transportation="公共交通", accommodation="经济型",
        ),
        messages=[],
        trip_plan=None,
        summary="",
        weather_outputs=[],
        hotel_outputs=[],
        poi_outputs=[],
    )


@pytest.mark.asyncio
async def test_save_and_load_session():
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("app.services.session_store._get_redis", return_value=fake):
        from app.services.session_store import save_session, load_session
        state = _make_state()
        await save_session("u1", state)
        loaded = await load_session("u1")

    assert loaded is not None
    assert loaded["trip_request"].city == "北京"


@pytest.mark.asyncio
async def test_load_nonexistent_session_returns_none():
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("app.services.session_store._get_redis", return_value=fake):
        from app.services.session_store import load_session
        result = await load_session("nonexistent")

    assert result is None


@pytest.mark.asyncio
async def test_save_refreshes_ttl():
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    with patch("app.services.session_store._get_redis", return_value=fake):
        from app.services.session_store import save_session
        state = _make_state()
        await save_session("u1", state)
        ttl = await fake.ttl("session:u1")

    assert ttl > 0
