import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.models.schemas import TripRequest, TripPlan, WeatherInfo, Hotel, Attraction, Location


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


@pytest.mark.asyncio
async def test_supervisor_returns_trip_plan():
    weather_out = [WeatherInfo(date="2025-06-01", day_weather="晴", night_weather="多云", day_temp=28, night_temp=18)]
    hotel_out = [Hotel(name="如家", address="北京朝阳", price_range="200-300", rating="4.2", type="经济型")]
    poi_out = [Attraction(name="故宫", address="东城区", location=Location(longitude=116.4, latitude=39.9), visit_duration=180, description="历史")]

    plan_json = '{"city":"北京","start_date":"2025-06-01","end_date":"2025-06-03","days":[],"overall_suggestions":"推荐早起"}'

    mock_plan = TripPlan(city="北京", start_date="2025-06-01", end_date="2025-06-03",
                         days=[], overall_suggestions="推荐早起")
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_plan)

    with patch("app.agents.supervisor.weather_subgraph") as mock_w, \
         patch("app.agents.supervisor.hotel_subgraph") as mock_h, \
         patch("app.agents.supervisor.poi_subgraph") as mock_p, \
         patch("app.agents.supervisor.get_structured_chain", return_value=mock_chain):

        mock_w.ainvoke = AsyncMock(return_value={"weather_result": weather_out})
        mock_h.ainvoke = AsyncMock(return_value={"hotel_result": hotel_out})
        mock_p.ainvoke = AsyncMock(return_value={"poi_result": poi_out})

        from app.agents.supervisor import supervisor_graph
        result = await supervisor_graph.ainvoke(_make_initial_state(_make_request()))

    assert result["trip_plan"] is not None
    assert result["trip_plan"].city == "北京"


@pytest.mark.asyncio
async def test_all_three_subgraphs_invoked():
    mock_plan = TripPlan(city="北京", start_date="2025-06-01", end_date="2025-06-03",
                         days=[], overall_suggestions="ok")
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_plan)

    with patch("app.agents.supervisor.weather_subgraph") as mock_w, \
         patch("app.agents.supervisor.hotel_subgraph") as mock_h, \
         patch("app.agents.supervisor.poi_subgraph") as mock_p, \
         patch("app.agents.supervisor.get_structured_chain", return_value=mock_chain):

        mock_w.ainvoke = AsyncMock(return_value={"weather_result": []})
        mock_h.ainvoke = AsyncMock(return_value={"hotel_result": []})
        mock_p.ainvoke = AsyncMock(return_value={"poi_result": []})

        from app.agents.supervisor import supervisor_graph
        await supervisor_graph.ainvoke(_make_initial_state(_make_request()))

    mock_w.ainvoke.assert_called_once()
    mock_h.ainvoke.assert_called_once()
    mock_p.ainvoke.assert_called_once()
