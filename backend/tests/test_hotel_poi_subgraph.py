import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.state import HotelSubState, POISubState


@pytest.mark.asyncio
async def test_hotel_subgraph_returns_hotel_list():
    mock_tool = MagicMock()
    mock_tool.name = "maps_text_search"
    mock_tool.ainvoke = AsyncMock(return_value='{"pois":[{"name":"如家酒店"}]}')

    llm_response = MagicMock(content='[{"name":"如家酒店","address":"北京市朝阳区","price_range":"200-300元/晚","rating":"4.2","distance":"市中心","type":"经济型","estimated_cost":250}]')

    with patch("app.agents.subgraphs.hotel.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.hotel.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):
            from app.agents.subgraphs.hotel import hotel_subgraph
            result = await hotel_subgraph.ainvoke(
                HotelSubState(city="北京", accommodation_pref="经济型", budget_level="mid", raw_result="", hotel_result=[])
            )

    assert len(result["hotel_result"]) == 1
    assert result["hotel_result"][0].name == "如家酒店"


@pytest.mark.asyncio
async def test_poi_subgraph_returns_attraction_list():
    mock_tool = MagicMock()
    mock_tool.name = "maps_text_search"
    mock_tool.ainvoke = AsyncMock(return_value='{"pois":[{"name":"故宫"}]}')

    llm_response = MagicMock(content='[{"name":"故宫","address":"北京市东城区景山前街4号","location":{"longitude":116.397,"latitude":39.916},"visit_duration":180,"description":"历史","category":"历史文化","rating":4.8,"ticket_price":60}]')

    with patch("app.agents.subgraphs.poi.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.poi.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):
            from app.agents.subgraphs.poi import poi_subgraph
            result = await poi_subgraph.ainvoke(
                POISubState(city="北京", preferences=["历史文化"], travel_days=3, raw_result="", poi_result=[])
            )

    assert len(result["poi_result"]) == 1
    assert result["poi_result"][0].name == "故宫"
