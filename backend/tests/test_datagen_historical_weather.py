from unittest.mock import patch
from datetime import date, timedelta
from app.models.schemas import TripRequest
from ml.planner.datagen.historical_weather import is_past_trip, fetch_historical_trip_weather


def _req(city="杭州", start="2020-04-01", days=3):
    end = (date.fromisoformat(start) + timedelta(days=days - 1)).isoformat()
    return TripRequest(city=city, start_date=start, end_date=end, travel_days=days,
                       transportation="打车", accommodation="经济型酒店", preferences=[],
                       party={"adults": 2}, budget_constraint={"amount": 3000, "strictness": "soft"})


def test_is_past_trip_true_for_old_dates():
    assert is_past_trip(_req(start="2020-04-01")) is True


def test_is_past_trip_false_for_future_dates():
    future = (date.today() + timedelta(days=30)).isoformat()
    assert is_past_trip(_req(start=future)) is False


def test_fetch_returns_empty_for_unknown_city():
    assert fetch_historical_trip_weather(_req(city="不存在城")) == []


def test_fetch_normalizes_open_meteo_daily():
    fake = {"daily": {"time": ["2020-04-01", "2020-04-02", "2020-04-03"],
                      "weather_code": [0, 61, 3],
                      "temperature_2m_max": [20.1, 18.0, 16.5],
                      "temperature_2m_min": [10.0, 9.2, 8.1]}}
    with patch("ml.planner.datagen.historical_weather.fetch_open_meteo_archive", return_value=fake):
        rows = fetch_historical_trip_weather(_req())
    assert len(rows) == 3
    assert rows[0]["day_weather"] == "晴"        # weather_code 0 → 晴
    assert rows[0]["source"] == "open_meteo_archive"
    assert int(rows[0]["day_temp"]) == 20
