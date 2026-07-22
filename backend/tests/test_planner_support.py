from app.models.schemas import TripRequest
from app.planner import dates, compact


def _req():
    return TripRequest(city="杭州", start_date="2026-08-01", end_date="2026-08-03",
                       travel_days=3, transportation="打车", accommodation="经济型酒店")


def test_trip_date_strings():
    assert dates.trip_date_strings(_req()) == ["2026-08-01", "2026-08-02", "2026-08-03"]


def test_compact_reduces_context():
    ctx = {"tool_snapshot": {"food_pois": [{"name": "x", "raw": "y" * 500}]}}
    out = compact.compact_for_planner(ctx)
    assert "tool_snapshot" in out
