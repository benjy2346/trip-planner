from app.models.schemas import (
    TripRequest, TripPlan, DayPlan, Attraction, Meal, Hotel, WeatherInfo, Location, Budget,
)
from app.planner.context import build_planner_context
from app.planner.validation import validate_trip_plan, recompute_budget


def _ctx():
    req = TripRequest(
        city="杭州", start_date="2026-08-01", end_date="2026-08-02",
        travel_days=2, transportation="打车", accommodation="经济型酒店",
        party={"adults": 2},
    )
    weather = [
        WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云", day_temp=30, night_temp=22),
        WeatherInfo(date="2026-08-02", day_weather="多云", night_weather="多云", day_temp=31, night_temp=23),
    ]
    hotels = [Hotel(name="如家杭州店", address="上城区", estimated_cost=250, type="经济型")]
    pois = [
        Attraction(name="西湖", address="西湖区", location=Location(longitude=120.1, latitude=30.2),
                   visit_duration=120, description="湖", ticket_price=0),
        Attraction(name="灵隐寺", address="西湖区", location=Location(longitude=120.10, latitude=30.24),
                   visit_duration=120, description="寺", ticket_price=30),
    ]
    return build_planner_context(req, weather, hotels, pois)


def _hotel():
    return Hotel(name="如家杭州店", address="上城区", estimated_cost=250, type="经济型", distance="")


def _meals(prefix):
    return [
        Meal(type="breakfast", name=f"{prefix}豆浆店", estimated_cost=15),
        Meal(type="lunch", name=f"{prefix}面馆", estimated_cost=40),
        Meal(type="dinner", name=f"{prefix}杭帮菜", estimated_cost=80),
    ]


def _good_plan():
    return TripPlan(
        city="杭州", start_date="2026-08-01", end_date="2026-08-02",
        days=[
            DayPlan(date="2026-08-01", day_index=0, description="d1", transportation="打车",
                    accommodation="经济型酒店", hotel=_hotel(),
                    attractions=[Attraction(name="西湖", address="西湖区",
                                            location=Location(longitude=120.1, latitude=30.2),
                                            visit_duration=120, description="湖", ticket_price=0)],
                    meals=_meals("知味观")),
            DayPlan(date="2026-08-02", day_index=1, description="d2", transportation="打车",
                    accommodation="经济型酒店", hotel=None,
                    attractions=[Attraction(name="灵隐寺", address="西湖区",
                                            location=Location(longitude=120.10, latitude=30.24),
                                            visit_duration=120, description="寺", ticket_price=30)],
                    meals=_meals("外婆家")),
        ],
        weather_info=[
            WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云", day_temp=30, night_temp=22),
            WeatherInfo(date="2026-08-02", day_weather="多云", night_weather="多云", day_temp=31, night_temp=23),
        ],
        overall_suggestions="ok",
        budget=Budget(total_transportation=200),
    )


def test_good_plan_passes():
    assert validate_trip_plan(_good_plan(), _ctx()) == []


def test_wrong_day_count_flagged():
    plan = _good_plan()
    plan.days = plan.days[:1]
    assert any("days" in v for v in validate_trip_plan(plan, _ctx()))


def test_missing_dinner_flagged():
    plan = _good_plan()
    plan.days[1].meals = plan.days[1].meals[:2]  # 去掉最后一天 dinner
    assert any("dinner" in v for v in validate_trip_plan(plan, _ctx()))


def test_placeholder_meal_flagged():
    plan = _good_plan()
    plan.days[0].meals[1].name = "附近餐厅"
    assert any("占位" in v for v in validate_trip_plan(plan, _ctx()))


def test_missing_hotel_on_lodging_day_flagged():
    plan = _good_plan()
    plan.days[0].hotel = None
    assert any("hotel" in v for v in validate_trip_plan(plan, _ctx()))


def test_fake_distance_flagged():
    plan = _good_plan()
    plan.days[0].hotel.distance = "距离景点2公里"
    assert any("distance" in v for v in validate_trip_plan(plan, _ctx()))


def test_ungrounded_attraction_flagged():
    plan = _good_plan()
    plan.days[0].attractions[0].name = "编造乐园"
    assert any("候选" in v for v in validate_trip_plan(plan, _ctx()))


def test_weather_mismatch_flagged():
    plan = _good_plan()
    plan.weather_info[0].day_weather = "暴雪"
    assert any("天气" in v for v in validate_trip_plan(plan, _ctx()))


def test_recompute_budget_units():
    b = recompute_budget(_good_plan(), party_total=2)
    assert b.total_hotels == 250            # 1 晚 × 250（按有 hotel 的天数）
    assert b.total_attractions == 60        # (0+30) × 2 人
    assert b.total_meals == 540             # (15+40+80)×2天 ×2人
    assert b.total_transportation == 200    # 沿用模型自报
    assert b.total == 250 + 60 + 540 + 200
