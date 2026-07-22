import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.models.schemas import TripRequest


def _make_request():
    return TripRequest(
        user_id="test-user",
        city="北京",
        start_date="2025-06-01",
        end_date="2025-06-03",
        travel_days=3,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )


def _make_initial_state(request):
    from app.agents.state import SupervisorState
    return SupervisorState(
        trip_request=request,
        messages=[],
        trip_plan=None,
        summary="",
        weather_outputs=[],
        hotel_outputs=[],
        poi_outputs=[],
    )


def _fake_context():
    """Builder.collect() 产物的最小 grounded 形状（helloagents 键名）。"""
    return {
        "request": {"city": "北京", "start_date": "2025-06-01", "end_date": "2025-06-03"},
        "party": {"total": 2},
        "preference_profile": {"diet_avoid": []},
        "planner_constraints": {"expected_dates": ["2025-06-01", "2025-06-02", "2025-06-03"]},
        "tool_snapshot": {
            "trip_weather": [],
            "classic_pois": [], "preference_pois": [], "scenic_pois": [], "experience_pois": [],
            "hotel_pois": [], "food_pois": [],
        },
    }


@pytest.mark.asyncio
async def test_supervisor_returns_trip_plan():
    plan_json = '{"city":"北京","start_date":"2025-06-01","end_date":"2025-06-03","days":[],"overall_suggestions":"推荐早起"}'
    mock_response = MagicMock()
    mock_response.content = plan_json

    with patch("app.agents.supervisor._planner_builder.collect", return_value=_fake_context()), \
         patch("app.agents.supervisor._planner_builder.compact_for_planner", side_effect=lambda c: c), \
         patch("app.agents.supervisor.acall_agent_with_fallback", AsyncMock(return_value=mock_response)):

        from app.agents.supervisor import create_supervisor_graph
        result = await create_supervisor_graph().ainvoke(_make_initial_state(_make_request()))

    assert result["trip_plan"] is not None
    assert result["trip_plan"].city == "北京"


@pytest.mark.asyncio
async def test_supervisor_uses_builder_and_validates():
    """取数走 Builder（不再是三个子图），校验走 grounded 校验、只告警不拦截。"""
    plan_json = '{"city":"北京","start_date":"2025-06-01","end_date":"2025-06-03","days":[],"overall_suggestions":"ok"}'
    mock_response = MagicMock()
    mock_response.content = plan_json

    with patch("app.agents.supervisor._planner_builder.collect", return_value=_fake_context()) as mock_collect, \
         patch("app.agents.supervisor._planner_builder.compact_for_planner", side_effect=lambda c: c), \
         patch("app.agents.supervisor.validate_grounded_trip_plan", return_value=["第1天 缺少 lunch"]) as mock_validate, \
         patch("app.agents.supervisor.acall_agent_with_fallback", AsyncMock(return_value=mock_response)):

        from app.agents.supervisor import create_supervisor_graph
        result = await create_supervisor_graph().ainvoke(_make_initial_state(_make_request()))

    mock_collect.assert_called_once()
    mock_validate.assert_called_once()
    # 校验有违规也不拦截，仍返回行程
    assert result["trip_plan"] is not None
    assert result["trip_plan"].city == "北京"
