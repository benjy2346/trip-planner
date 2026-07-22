"""grounded 打分器单测。夹具与 test_planner_validation_grounded.py 同源：
compact_planner_context 作为打分上下文，plan 序列化成 JSON 文本喂给 evaluate_output。"""
import json

from app.models.schemas import (
    Attraction, Budget, DayPlan, Hotel, Location, Meal, TripPlan,
)
from ml.planner.eval.rule_metrics import aggregate, evaluate_output


def _loc(lng=120.1, lat=30.2):
    return Location(longitude=lng, latitude=lat)


def _ctx(days=2, food=("外婆家", "知味观", "新白鹿", "绿茶餐厅"),
         diet_avoid=(), amount=3000, strictness="soft"):
    dates = [f"2026-08-0{i + 1}" for i in range(days)]
    return {
        "record_id": "planner_standard200_realbudget_eval_000000",
        "split": "eval",
        "compact_planner_context": {
            "request": {"city": "杭州", "start_date": dates[0], "end_date": dates[-1],
                        "travel_days": days, "accommodation": "经济型酒店"},
            "party": {"total": 2},
            "budget_constraint": {"amount": amount, "strictness": strictness},
            "preference_profile": {"diet_avoid": list(diet_avoid)},
            "planner_constraints": {"days_count": days, "expected_dates": dates},
            "tool_snapshot": {
                "trip_weather": [],
                "classic_pois": [{"name": "西湖", "location": {"longitude": 120.1, "latitude": 30.2}}],
                "preference_pois": [{"name": "河坊街", "location": {"longitude": 120.2, "latitude": 30.3}}],
                "scenic_pois": [{"name": "灵隐寺", "location": {"longitude": 120.3, "latitude": 30.4}}],
                "experience_pois": [{"name": "宋城", "location": {"longitude": 120.4, "latitude": 30.5}}],
                "hotel_pois": [{"name": "如家酒店", "location": {"longitude": 120.15, "latitude": 30.25}}],
                "food_pois": [{"name": n, "location": {"longitude": 120.1, "latitude": 30.2}} for n in food],
            },
        },
    }


def _meals(b="酒店早餐", l="知味观", d="新白鹿"):
    return [Meal(type="breakfast", name=b, location=_loc(), estimated_cost=30),
            Meal(type="lunch", name=l, location=_loc(), estimated_cost=60),
            Meal(type="dinner", name=d, location=_loc(), estimated_cost=80)]


def _day(idx, date, attractions=("西湖",), meals=None, hotel="如家酒店"):
    return DayPlan(
        date=date, day_index=idx, description="d", transportation="打车",
        accommodation="经济型酒店",
        hotel=None if hotel is None else Hotel(
            name=hotel, address="x", location=_loc(), distance="", estimated_cost=400),
        attractions=[Attraction(name=n, address="x", location=_loc(), visit_duration=120,
                                description="l", ticket_price=40) for n in attractions],
        meals=meals if meals is not None else _meals())


def _two_day_plan_json(**day2):
    d1 = _day(0, "2026-08-01", attractions=("西湖",), meals=_meals())
    defaults = dict(attractions=("灵隐寺",), meals=_meals(l="绿茶餐厅", d="外婆家"), hotel=None)
    defaults.update(day2)
    d2 = _day(1, "2026-08-02", **defaults)
    plan = TripPlan(city="杭州", start_date="2026-08-01", end_date="2026-08-02",
                    days=[d1, d2], weather_info=[], overall_suggestions="ok",
                    budget=Budget(total_transportation=200))
    return plan.model_dump_json()


def test_clean_plan_hard_passes():
    m = evaluate_output(_ctx(), _two_day_plan_json())
    assert m["json_ok"] and m["schema_ok"]
    assert m["hard_pass"] is True
    assert m["violations"] == []


def test_ungrounded_attraction_lowers_hard_pass_and_grounding():
    m = evaluate_output(_ctx(), _two_day_plan_json(attractions=("不存在的景点",)))
    assert m["hard_pass"] is False
    assert any("景点" in x and "候选" in x for x in m["violations"])
    # 第2天景点不在候选：2 个景点里命中 1 个
    assert m["attraction_grounding_rate"] == 50.0


def test_meal_grounding_excludes_lodging_breakfast():
    # 酒店早餐不计入 grounding 分母；4 个午晚餐全部命中
    m = evaluate_output(_ctx(), _two_day_plan_json())
    assert m["meal_grounding_rate"] == 100.0


def test_same_day_lunch_dinner_repeat_counted():
    m = evaluate_output(_ctx(), _two_day_plan_json(meals=_meals(l="外婆家", d="外婆家")))
    assert m["meal_repeat_count"] >= 1
    assert m["soft_pass"] is False


def test_hard_budget_overspend_flagged():
    # 硬预算 amount=100，重算 total 远超 → budget_ok False
    m = evaluate_output(_ctx(amount=100, strictness="hard"), _two_day_plan_json())
    assert m["budget_ok"] is False
    assert m["soft_pass"] is False


def test_no_budget_amount_is_ok():
    m = evaluate_output(_ctx(amount=0), _two_day_plan_json())
    assert m["budget_ok"] is True


def test_parse_failure_reported():
    m = evaluate_output(_ctx(), "这不是 JSON")
    assert m["json_ok"] is False
    assert m["hard_pass"] is False
    assert m["violations"] and m["violations"][0].startswith("parse:")


def test_aggregate_reports_rates():
    good = evaluate_output(_ctx(), _two_day_plan_json())
    bad = evaluate_output(_ctx(), "垃圾")
    agg = aggregate([good, bad])
    assert agg["count"] == 2
    assert agg["hard_pass"] == 50.0
    assert agg["json_ok"] == 50.0
