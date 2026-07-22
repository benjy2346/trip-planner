import json
import importlib.util
from pathlib import Path
from app.models.schemas import TripRequest, WeatherInfo, Hotel, Attraction, Location
from app.planner.context import build_planner_context

_spec = importlib.util.spec_from_file_location(
    "rule_eval", Path(__file__).resolve().parent.parent / "ml" / "planner" / "rule_eval.py")
rule_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rule_eval)


def _record():
    req = TripRequest(
        city="杭州", start_date="2026-08-01", end_date="2026-08-02", travel_days=2,
        transportation="打车", accommodation="经济型酒店", party={"adults": 2},
        budget_constraint={"amount": 2000, "strictness": "hard"},
    )
    ctx = build_planner_context(
        req,
        [WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云", day_temp=30, night_temp=22),
         WeatherInfo(date="2026-08-02", day_weather="多云", night_weather="多云", day_temp=31, night_temp=23)],
        [Hotel(name="如家杭州店", address="上城区", estimated_cost=250, type="经济型")],
        [Attraction(name="西湖", address="西湖区", location=Location(longitude=120.1, latitude=30.2),
                    visit_duration=120, description="湖", ticket_price=0),
         Attraction(name="灵隐寺", address="西湖区", location=Location(longitude=120.10, latitude=30.24),
                    visit_duration=120, description="寺", ticket_price=30)],
    )
    return {"record_id": "t_0001", "difficulty": "standard",
            "request": req.model_dump(), "context": ctx}


def _good_output():
    def day(i, d, hotel, attraction, ticket, m):
        return {
            "date": d, "day_index": i, "description": f"d{i+1}", "transportation": "打车",
            "accommodation": "经济型酒店", "hotel": hotel,
            "attractions": [{"name": attraction, "address": "西湖区",
                             "location": {"longitude": 120.1, "latitude": 30.2},
                             "visit_duration": 120, "description": "x", "ticket_price": ticket}],
            "meals": [
                {"type": "breakfast", "name": f"{m}豆浆店", "estimated_cost": 15},
                {"type": "lunch", "name": f"{m}面馆", "estimated_cost": 40},
                {"type": "dinner", "name": f"{m}杭帮菜", "estimated_cost": 80},
            ],
        }
    hotel = {"name": "如家杭州店", "address": "上城区", "distance": "", "type": "经济型",
             "estimated_cost": 250}
    return json.dumps({
        "city": "杭州", "start_date": "2026-08-01", "end_date": "2026-08-02",
        "days": [day(0, "2026-08-01", hotel, "西湖", 0, "知味观"),
                 day(1, "2026-08-02", None, "灵隐寺", 30, "外婆家")],
        "weather_info": [
            {"date": "2026-08-01", "day_weather": "晴", "night_weather": "多云", "day_temp": 30, "night_temp": 22},
            {"date": "2026-08-02", "day_weather": "多云", "night_weather": "多云", "day_temp": 31, "night_temp": 23}],
        "overall_suggestions": "ok",
        "budget": {"total_attractions": 60, "total_hotels": 250, "total_meals": 540,
                   "total_transportation": 200, "total": 1050},
    }, ensure_ascii=False)


def test_good_output_hard_and_soft_pass():
    m = rule_eval.evaluate_output(_record(), _good_output())
    assert m["json_ok"] and m["schema_ok"]
    assert m["hard_pass"] is True
    assert m["meal_repeat_count"] == 0
    assert m["budget_ok"] is True
    assert m["soft_pass"] is True


def test_broken_json_fails_hard():
    m = rule_eval.evaluate_output(_record(), "{这不是json")
    assert m["json_ok"] is False and m["hard_pass"] is False


def test_repeated_dinner_fails_soft_only():
    out = json.loads(_good_output())
    out["days"][1]["meals"][2]["name"] = out["days"][0]["meals"][2]["name"]
    m = rule_eval.evaluate_output(_record(), json.dumps(out, ensure_ascii=False))
    assert m["hard_pass"] is True
    assert m["meal_repeat_count"] == 1
    assert m["soft_pass"] is False


def test_over_budget_fails_soft():
    out = json.loads(_good_output())
    out["days"][0]["hotel"]["estimated_cost"] = 5000
    m = rule_eval.evaluate_output(_record(), json.dumps(out, ensure_ascii=False))
    assert m["budget_ok"] is False and m["soft_pass"] is False
