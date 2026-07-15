from datetime import date
from app.models.schemas import TripRequest
from ml.planner.datagen.requests import iter_requests, to_trip_request


def test_seed_reproducible():
    a = iter_requests(10, seed=9200, date_mode="mixed")
    b = iter_requests(10, seed=9200, date_mode="mixed")
    assert [x["city"] for x in a] == [x["city"] for x in b]


def test_control_spec_present():
    items = iter_requests(5, seed=9200, date_mode="mixed")
    for it in items:
        cs = it["control_spec"]
        for k in ("city_tier", "companion_type", "budget_level", "budget_strictness"):
            assert k in cs


def test_mixed_mode_includes_past_dates():
    items = iter_requests(40, seed=9200, date_mode="mixed")
    today = date.today()
    assert any(date.fromisoformat(it["end_date"]) < today for it in items)


def test_to_trip_request_valid():
    item = iter_requests(1, seed=9200, date_mode="mixed")[0]
    req = to_trip_request(item)
    assert isinstance(req, TripRequest)
    assert req.travel_days >= 1
