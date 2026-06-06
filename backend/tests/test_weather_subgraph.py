import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.state import WeatherSubState
from app.models.schemas import WeatherInfo


def _mock_structured_chain(return_obj):
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=return_obj)
    return chain


@pytest.mark.asyncio
async def test_weather_subgraph_returns_weather_info():
    mock_tool = MagicMock()
    mock_tool.name = "maps_weather"
    mock_tool.ainvoke = AsyncMock(return_value='{"status":"1"}')

    weather_obj = MagicMock()
    weather_obj.weather_result = [
        WeatherInfo(date="2025-06-01", day_weather="晴", night_weather="多云",
                    day_temp=28, night_temp=18, wind_direction="南", wind_power="3级")
    ]

    with patch("app.agents.subgraphs.weather.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.weather.get_structured_chain",
                   return_value=_mock_structured_chain(weather_obj)):
            from app.agents.subgraphs.weather import weather_subgraph
            result = await weather_subgraph.ainvoke(
                WeatherSubState(city="北京", travel_dates=["2025-06-01"],
                                raw_result="", weather_result=[])
            )

    assert len(result["weather_result"]) == 1
    assert result["weather_result"][0].day_weather == "晴"


@pytest.mark.asyncio
async def test_weather_subgraph_no_tool_returns_empty():
    weather_obj = MagicMock()
    weather_obj.weather_result = []

    with patch("app.agents.subgraphs.weather.get_amap_tools", return_value=[]):
        with patch("app.agents.subgraphs.weather.get_structured_chain",
                   return_value=_mock_structured_chain(weather_obj)):
            from app.agents.subgraphs.weather import weather_subgraph
            result = await weather_subgraph.ainvoke(
                WeatherSubState(city="北京", travel_dates=["2025-06-01"],
                                raw_result="", weather_result=[])
            )

    assert result["weather_result"] == []
