import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.state import SupervisorState
from app.models.schemas import (
    TripPlan, DayPlan, Hotel, Attraction, Meal, WeatherInfo, Budget, Location
)


def _make_state(user_message: str) -> SupervisorState:
    hotel = Hotel(
        name="北京假日酒店", address="长安街1号",
        location=Location(longitude=116.4, latitude=39.9),
        price_range="300-500元", rating="4.5", distance="1km",
        type="经济型", estimated_cost=400,
    )
    attraction = Attraction(
        name="故宫", address="北京市东城区",
        location=Location(longitude=116.397, latitude=39.916),
        visit_duration=180, description="明清皇宫，世界文化遗产",
        category="历史文化", ticket_price=60,
    )
    meal = Meal(type="lunch", name="全聚德", description="北京烤鸭", estimated_cost=80)
    day = DayPlan(
        date="2025-06-01", day_index=0, description="游览故宫",
        transportation="地铁", accommodation="经济型",
        hotel=hotel, attractions=[attraction], meals=[meal],
    )
    weather = WeatherInfo(
        date="2025-06-01", day_weather="晴", night_weather="多云",
        day_temp=28, night_temp=18, wind_direction="南风", wind_power="1-3级",
    )
    budget = Budget(
        total_attractions=60, total_hotels=400,
        total_meals=80, total_transportation=50, total=590,
    )
    plan = TripPlan(
        city="北京", start_date="2025-06-01", end_date="2025-06-01",
        days=[day], weather_info=[weather], overall_suggestions="带好遮阳帽",
        budget=budget,
    )
    from app.models.schemas import TripRequest
    return SupervisorState(
        trip_request=TripRequest(
            user_id="u1", city="北京", start_date="2025-06-01",
            end_date="2025-06-01", travel_days=1,
            transportation="地铁", accommodation="经济型",
        ),
        messages=[HumanMessage(content=user_message)],
        trip_plan=plan,
        summary="",
        weather_outputs=[],
        hotel_outputs=[],
        poi_outputs=[],
    )


@pytest.mark.asyncio
async def test_query_handler_hotel():
    from app.agents.chat_graph import query_handler_node
    state = _make_state("第一天住哪个酒店")
    result = await query_handler_node(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "北京假日酒店" in result["messages"][0].content


@pytest.mark.asyncio
async def test_query_handler_budget():
    from app.agents.chat_graph import query_handler_node
    state = _make_state("总费用是多少")
    result = await query_handler_node(state)
    assert "590" in result["messages"][0].content


@pytest.mark.asyncio
async def test_query_handler_weather():
    from app.agents.chat_graph import query_handler_node
    state = _make_state("天气怎么样")
    result = await query_handler_node(state)
    assert "晴" in result["messages"][0].content


@pytest.mark.asyncio
async def test_query_handler_no_plan():
    from app.agents.chat_graph import query_handler_node
    from app.models.schemas import TripRequest
    state = SupervisorState(
        trip_request=TripRequest(
            user_id="u1", city="北京", start_date="2025-06-01",
            end_date="2025-06-01", travel_days=1,
            transportation="地铁", accommodation="经济型",
        ),
        messages=[HumanMessage(content="第一天住哪")],
        trip_plan=None,
        summary="", weather_outputs=[], hotel_outputs=[], poi_outputs=[],
    )
    result = await query_handler_node(state)
    assert "还没有生成行程" in result["messages"][0].content


@pytest.mark.asyncio
async def test_other_handler_returns_canned():
    from app.agents.chat_graph import other_handler_node
    state = _make_state("谢谢")
    result = await other_handler_node(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "行程助手" in result["messages"][0].content


@pytest.mark.asyncio
async def test_modify_handler_calls_llm_and_returns_updated_plan():
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.agents.chat_graph import modify_handler_node

    mock_response = MagicMock()
    mock_response.content = '{"reply": "已修改行程", "updated_plan": null}'
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("app.agents.chat_graph.get_agent_llm", return_value=mock_llm):
        state = _make_state("帮我改轻松一点")
        result = await modify_handler_node(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "已修改行程"


@pytest.mark.asyncio
async def test_classify_intent_node_routes_to_query():
    from unittest.mock import AsyncMock, patch
    from app.agents.chat_graph import classify_intent_node

    with patch("app.agents.chat_graph.classify_intent", AsyncMock(return_value="query_plan")):
        state = _make_state("第一天住哪")
        cmd = await classify_intent_node(state)

    assert cmd.goto == "query_handler"


@pytest.mark.asyncio
async def test_classify_intent_node_routes_to_modify():
    from unittest.mock import AsyncMock, patch
    from app.agents.chat_graph import classify_intent_node

    with patch("app.agents.chat_graph.classify_intent", AsyncMock(return_value="modify")):
        state = _make_state("帮我改一下")
        cmd = await classify_intent_node(state)

    assert cmd.goto == "modify_handler"
