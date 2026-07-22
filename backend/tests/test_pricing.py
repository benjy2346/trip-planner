from app.planner.pricing_legacy import hotel_price, meal_cost_table, city_tier
from app.models.schemas import TripRequest, WeatherInfo, Hotel, Attraction, Location
from app.planner.context import build_planner_context


def test_city_tier_known_and_default():
    assert city_tier("北京") == 1
    assert city_tier("杭州") == 2
    assert city_tier("未知城市") == 2


def test_hotel_price_monotonic_by_tier():
    # 高档 > 舒适 > 经济
    assert (hotel_price("高档型酒店", "杭州") > hotel_price("舒适型酒店", "杭州")
            > hotel_price("经济型酒店", "杭州"))
    # 一线城市更贵
    assert hotel_price("经济型酒店", "北京") > hotel_price("经济型酒店", "杭州")


def test_hotel_price_has_candidate_spread():
    prices = {hotel_price("舒适型酒店", "杭州", i) for i in range(3)}
    assert len(prices) >= 2  # 候选之间有价差，便于"挑便宜的"


def test_meal_cost_table_positive_and_typed():
    t = meal_cost_table("北京")
    assert set(t) == {"breakfast", "lunch", "dinner"}
    assert t["dinner"] > t["breakfast"] > 0


def test_context_injects_hotel_price_when_missing():
    req = TripRequest(city="北京", start_date="2026-08-01", end_date="2026-08-02",
                      travel_days=2, transportation="打车", accommodation="舒适型酒店")
    weather = [WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云",
                           day_temp=30, night_temp=22)]
    hotels = [Hotel(name="A酒店", address="x", estimated_cost=0, type="舒适型")]  # 地图无价
    pois = [Attraction(name="故宫", address="x", location=Location(longitude=1, latitude=1),
                       visit_duration=120, description="x", ticket_price=60)]
    ctx = build_planner_context(req, weather, hotels, pois)
    injected = ctx["tool_snapshot"]["hotel_candidates"][0]["estimated_cost"]
    assert injected == hotel_price("舒适型酒店", "北京", 0)
    assert injected > 0
    assert ctx["pricing_policy"]["meal_cost_standard"]["dinner"] > 0


def test_context_keeps_existing_hotel_price():
    req = TripRequest(city="北京", start_date="2026-08-01", end_date="2026-08-02",
                      travel_days=2, transportation="打车", accommodation="经济型酒店")
    hotels = [Hotel(name="B酒店", address="x", estimated_cost=333, type="经济型")]
    pois = [Attraction(name="故宫", address="x", location=Location(longitude=1, latitude=1),
                       visit_duration=120, description="x", ticket_price=0)]
    ctx = build_planner_context(req, [], hotels, pois)
    assert ctx["tool_snapshot"]["hotel_candidates"][0]["estimated_cost"] == 333
