from unittest.mock import patch
from datetime import date, timedelta
from app.models.schemas import TripRequest
from ml.planner.datagen.context_builder import DataGenPlannerContextBuilder


def _req(start, days=3):
    end = (date.fromisoformat(start) + timedelta(days=days - 1)).isoformat()
    return TripRequest(city="杭州", start_date=start, end_date=end, travel_days=days,
                       transportation="打车", accommodation="经济型酒店", preferences=[],
                       party={"adults": 2}, budget_constraint={"amount": 3000, "strictness": "soft"})


def test_past_trip_uses_open_meteo():
    b = DataGenPlannerContextBuilder(amap_api_key="X", historical_weather_provider="open-meteo")
    rows = [{"date": "2020-04-01", "day_weather": "晴", "source": "open_meteo_archive"}]
    with patch("ml.planner.datagen.context_builder.fetch_historical_trip_weather", return_value=rows), \
         patch("ml.planner.datagen.context_builder.throttle_open_meteo_call"):
        snap = b._collect_weather_snapshot(_req("2020-04-01"))
    assert snap["tool_snapshot"]["trip_weather"] == rows


def test_future_trip_falls_back_to_super():
    b = DataGenPlannerContextBuilder(amap_api_key="X", historical_weather_provider="open-meteo")
    future = (date.today() + timedelta(days=30)).isoformat()
    with patch("app.planner.context.PlannerContextBuilder._collect_weather_snapshot",
               return_value={"tool_snapshot": {"trip_weather": []}, "status": {"ok": True, "message": "amap"}}) as sup:
        b._collect_weather_snapshot(_req(future))
    sup.assert_called_once()
