import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.state import WeatherSubState


@pytest.mark.asyncio
async def test_weather_subgraph_returns_weather_info():
    mock_tool = MagicMock()
    mock_tool.name = "maps_weather"
    mock_tool.ainvoke = AsyncMock(return_value='{"status":"1","forecasts":[{"city":"北京"}]}')

    llm_response = MagicMock(content='[{"date":"2025-06-01","day_weather":"晴","night_weather":"多云","day_temp":28,"night_temp":18,"wind_direction":"南","wind_power":"3级"}]')

    with patch("app.agents.subgraphs.weather.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.weather.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):
            from app.agents.subgraphs.weather import weather_subgraph
            result = await weather_subgraph.ainvoke(
                WeatherSubState(city="北京", travel_dates=["2025-06-01"], raw_result="", weather_result=[])
            )

    assert len(result["weather_result"]) == 1
    assert result["weather_result"][0].day_weather == "晴"


@pytest.mark.asyncio
async def test_weather_subgraph_handles_parse_error():
    mock_tool = MagicMock()
    mock_tool.name = "maps_weather"
    mock_tool.ainvoke = AsyncMock(return_value="error response")

    llm_response = MagicMock(content="invalid json {{")

    with patch("app.agents.subgraphs.weather.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.weather.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):
            from app.agents.subgraphs.weather import weather_subgraph
            result = await weather_subgraph.ainvoke(
                WeatherSubState(city="北京", travel_dates=["2025-06-01"], raw_result="", weather_result=[])
            )

    assert result["weather_result"] == []
