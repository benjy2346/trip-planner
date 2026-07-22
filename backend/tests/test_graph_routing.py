"""统一图的入口分流测试。

生成与对话此前是两张图，靠共用 checkpointer 隐式串联；合并后由 mode 显式分流，
这里锁住分流行为，避免以后改动把两条路径接错。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.graph import MODE_CHAT, MODE_GENERATE, create_trip_graph, entry_router_node
from app.models.schemas import TripRequest


def _request():
    return TripRequest(
        user_id="user-routing-test",
        city="北京",
        start_date="2025-06-01",
        end_date="2025-06-03",
        travel_days=3,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )


def _state(mode=None, **extra):
    from app.agents.state import SupervisorState

    state = SupervisorState(
        trip_request=_request(),
        trip_plan=None,
        summary="",
        weather_outputs=[],
        hotel_outputs=[],
        poi_outputs=[],
        **{"messages": [], **extra},
    )
    if mode is not None:
        state["mode"] = mode
    return state


@pytest.mark.asyncio
async def test_generate_mode_routes_to_assembler():
    command = await entry_router_node(_state(mode=MODE_GENERATE))
    assert command.goto == "assembler"


@pytest.mark.asyncio
async def test_chat_mode_routes_to_classify_intent():
    command = await entry_router_node(_state(mode=MODE_CHAT))
    assert command.goto == "classify_intent"


@pytest.mark.asyncio
async def test_missing_mode_defaults_to_chat():
    """缺省按对话处理：生成必须由调用方显式声明，避免误触发取数与 LLM 生成。"""
    command = await entry_router_node(_state())
    assert command.goto == "classify_intent"


@pytest.mark.asyncio
async def test_generate_mode_reaches_assembler_end_to_end():
    """整图跑通：mode=generate 应产出 trip_plan，不经过意图分类。"""
    plan_json = (
        '{"city":"北京","start_date":"2025-06-01","end_date":"2025-06-03",'
        '"days":[],"overall_suggestions":"ok"}'
    )
    response = MagicMock()
    response.content = plan_json
    context = {
        "request": {"city": "北京", "start_date": "2025-06-01", "end_date": "2025-06-03"},
        "party": {"total": 2},
        "preference_profile": {"diet_avoid": []},
        "planner_constraints": {"expected_dates": ["2025-06-01", "2025-06-02", "2025-06-03"]},
        "tool_snapshot": {
            "trip_weather": [], "classic_pois": [], "preference_pois": [],
            "scenic_pois": [], "experience_pois": [], "hotel_pois": [], "food_pois": [],
        },
    }

    with patch("app.agents.supervisor._planner_builder.collect", return_value=context), \
         patch("app.agents.supervisor._planner_builder.compact_for_planner", side_effect=lambda c: c), \
         patch("app.agents.supervisor.acall_agent_with_fallback", AsyncMock(return_value=response)), \
         patch("app.agents.chat_graph.classify_intent", AsyncMock()) as mock_classify:

        result = await create_trip_graph().ainvoke(_state(mode=MODE_GENERATE))

    assert result["trip_plan"] is not None
    mock_classify.assert_not_called()


@pytest.mark.asyncio
async def test_chat_mode_reaches_intent_classifier():
    """mode=chat 应走意图分类，不触发取数。"""
    from langchain_core.messages import HumanMessage

    with patch("app.agents.chat_graph.classify_intent", AsyncMock(return_value="other")) as mock_classify, \
         patch("app.agents.chat_graph.get_agent_llm") as mock_llm, \
         patch("app.agents.supervisor._planner_builder.collect") as mock_collect:
        mock_llm.return_value.ainvoke = AsyncMock(return_value=MagicMock(content="好的"))

        await create_trip_graph().ainvoke(
            _state(mode=MODE_CHAT, messages=[HumanMessage(content="你好")])
        )

    mock_classify.assert_awaited_once()
    mock_collect.assert_not_called()
