"""Grounded TripPlan 校验（Task 7）。

判定语义对齐 helloagents `training/scripts/eval/eval_rule_metrics.py::evaluate_output`
与 `backend/app/planner/output.py` 的 grounding 谓词，这样后端线上告警与训练/评测
的 hardpass 用的是同一套规则。

三个容易踩的点（我们旧 validation 与他不一致的地方）：
- 景点候选池是 classic/preference/scenic/experience **四个 bucket 的并集**；
- 名称匹配是**模糊别名匹配**（「锦里古街」命中候选「锦里」），不是精确相等；
- **酒店早餐是合法早餐来源**，不算占位词，也不计入餐厅重复次数。
"""
import math

import pytest

from app.models.schemas import (
    Attraction, Budget, DayPlan, Hotel, Location, Meal, TripPlan, WeatherInfo,
)
from app.planner.validation import recompute_grounded_budget, validate_grounded_trip_plan


def _loc(lng=120.1, lat=30.2):
    return Location(longitude=lng, latitude=lat)


def _ctx(days=2, food=("外婆家", "知味观", "新白鹿", "绿茶餐厅"), diet_avoid=("辣",)):
    dates = [f"2026-08-0{i + 1}" for i in range(days)]
    return {
        "request": {"city": "杭州", "start_date": dates[0], "end_date": dates[-1],
                    "travel_days": days, "accommodation": "经济型酒店"},
        "party": {"total": 2},
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
        meals=meals if meals is not None else _meals(),
    )


def _plan(days_list, city="杭州"):
    dates = [d.date for d in days_list]
    return TripPlan(city=city, start_date=dates[0], end_date=dates[-1], days=days_list,
                    weather_info=[], overall_suggestions="ok", budget=Budget())


def _two_day_plan(**day2_kwargs):
    """第1天住店、第2天(末日)退房的标准 2 天计划。"""
    d1 = _day(0, "2026-08-01", attractions=("西湖",), meals=_meals())
    defaults = dict(attractions=("灵隐寺",), meals=_meals(l="绿茶餐厅", d="外婆家"), hotel=None)
    defaults.update(day2_kwargs)
    d2 = _day(1, "2026-08-02", **defaults)
    return _plan([d1, d2])


# --- baseline ---------------------------------------------------------------

def test_clean_plan_has_no_violations():
    assert validate_grounded_trip_plan(_two_day_plan(), _ctx()) == []


# --- grounding --------------------------------------------------------------

def test_attraction_from_any_of_four_buckets_is_grounded():
    """experience_pois 里的候选也算 grounded，不能只认 scenic_pois。"""
    plan = _two_day_plan(attractions=("宋城",))
    assert validate_grounded_trip_plan(plan, _ctx()) == []


def test_fuzzy_alias_match_is_grounded():
    """「西湖风景名胜区」应命中候选「西湖」，不能按精确相等判违规。"""
    plan = _two_day_plan(attractions=("西湖风景名胜区",))
    assert validate_grounded_trip_plan(plan, _ctx()) == []


def test_ungrounded_attraction_flagged():
    v = validate_grounded_trip_plan(_two_day_plan(attractions=("不存在的景点",)), _ctx())
    assert any("景点" in x and "候选" in x for x in v)


def test_ungrounded_meal_flagged():
    v = validate_grounded_trip_plan(_two_day_plan(meals=_meals(l="不存在饭店")), _ctx())
    assert any("餐饮" in x and "候选" in x for x in v)


def test_lodging_breakfast_is_allowed():
    """酒店早餐不在 food_pois 里，但它是合法早餐来源，不该判违规。"""
    plan = _two_day_plan(meals=_meals(b="酒店早餐", l="绿茶餐厅", d="外婆家"))
    assert validate_grounded_trip_plan(plan, _ctx()) == []


def test_lodging_breakfast_only_valid_for_breakfast():
    """同样的名字放在晚餐位上就是没 grounding。"""
    v = validate_grounded_trip_plan(_two_day_plan(meals=_meals(d="酒店早餐")), _ctx())
    assert any("餐饮" in x for x in v)


def test_placeholder_meal_flagged():
    v = validate_grounded_trip_plan(_two_day_plan(meals=_meals(d="当地晚餐")), _ctx())
    assert any("餐饮" in x for x in v)


# --- diversity --------------------------------------------------------------

def test_same_day_lunch_dinner_repeat_flagged():
    v = validate_grounded_trip_plan(_two_day_plan(meals=_meals(l="外婆家", d="外婆家")), _ctx())
    assert any("午晚餐重复" in x for x in v)


def test_brand_suffix_counts_as_same_restaurant():
    """「外婆家(西湖店)」与「外婆家」在多样性上是同一家。"""
    ctx = _ctx(food=("外婆家", "外婆家(西湖店)", "知味观", "新白鹿"))
    v = validate_grounded_trip_plan(
        _two_day_plan(meals=_meals(l="外婆家", d="外婆家(西湖店)")), ctx)
    assert any("午晚餐重复" in x for x in v)


def test_meal_repeat_cap_flagged():
    """5 天行程，同一家店出现次数超过 max(2, (5*3+4)//5)=3 次即违规。"""
    dates = [f"2026-08-0{i + 1}" for i in range(5)]
    days = []
    for i, date in enumerate(dates):
        # 外婆家 连续 4 天当午餐 —— 超过上限 3
        meals = _meals(b="酒店早餐", l="外婆家", d="知味观") if i < 4 else _meals(b="酒店早餐", l="新白鹿", d="绿茶餐厅")
        days.append(_day(i, date, meals=meals, hotel=None if i == 4 else "如家酒店"))
    v = validate_grounded_trip_plan(_plan(days), _ctx(days=5))
    assert any("重复" in x and "外婆家" in x for x in v)


def test_attraction_repeat_across_days_flagged():
    plan = _two_day_plan(attractions=("西湖",))  # 第1天也是西湖
    v = validate_grounded_trip_plan(plan, _ctx())
    assert any("景点" in x and "重复" in x for x in v)


# --- dietary ----------------------------------------------------------------

def test_diet_avoid_meal_flagged():
    ctx = _ctx(food=("外婆家", "知味观", "新白鹿", "麻辣香锅"))
    v = validate_grounded_trip_plan(_two_day_plan(meals=_meals(d="麻辣香锅")), ctx)
    assert any("饮食约束" in x for x in v)


# --- structure / coords -----------------------------------------------------

def test_missing_coordinates_flagged():
    """旧 hollow context 的 0,0 占位坐标必须被抓出来。"""
    d1 = _day(0, "2026-08-01")
    d1.attractions[0].location = Location(longitude=0, latitude=0)
    d2 = _day(1, "2026-08-02", attractions=("灵隐寺",),
              meals=_meals(l="绿茶餐厅", d="外婆家"), hotel=None)
    v = validate_grounded_trip_plan(_plan([d1, d2]), _ctx())
    assert any("坐标" in x for x in v)


def test_invalid_hotel_name_flagged():
    v = validate_grounded_trip_plan(_two_day_plan(), _ctx())
    assert v == []
    d1 = _day(0, "2026-08-01", hotel="返程")
    d2 = _day(1, "2026-08-02", attractions=("灵隐寺",),
              meals=_meals(l="绿茶餐厅", d="外婆家"), hotel=None)
    v = validate_grounded_trip_plan(_plan([d1, d2]), _ctx())
    assert any("酒店" in x for x in v)


def test_placeholder_hotel_distance_flagged():
    d1 = _day(0, "2026-08-01")
    d1.hotel.distance = "距离景点2公里"
    d2 = _day(1, "2026-08-02", attractions=("灵隐寺",),
              meals=_meals(l="绿茶餐厅", d="外婆家"), hotel=None)
    v = validate_grounded_trip_plan(_plan([d1, d2]), _ctx())
    assert any("distance" in x or "距离" in x for x in v)


def test_middle_day_missing_hotel_flagged():
    d1 = _day(0, "2026-08-01", hotel=None)
    d2 = _day(1, "2026-08-02", attractions=("灵隐寺",),
              meals=_meals(l="绿茶餐厅", d="外婆家"), hotel=None)
    v = validate_grounded_trip_plan(_plan([d1, d2]), _ctx())
    assert any("hotel" in x or "住宿" in x for x in v)


# --- budget -----------------------------------------------------------------

@pytest.mark.parametrize("party_total,rooms", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)])
def test_rooms_is_two_people_per_room(party_total, rooms):
    assert rooms == math.ceil(party_total / 2)
    plan = _two_day_plan()
    plan.budget = Budget(total_transportation=200)
    b = recompute_grounded_budget(plan, party_total)
    # 只有第1天有酒店（末日退房）：400 * rooms
    assert b.total_hotels == 400 * rooms


def test_budget_multiplies_tickets_and_meals_by_party():
    plan = _two_day_plan()
    plan.budget = Budget(total_transportation=200)
    b = recompute_grounded_budget(plan, party_total=2)

    assert b.total_attractions == 40 * 2 * 2            # 2 天 × 1 个 40 元景点 × 2 人
    assert b.total_meals == (30 + 60 + 80) * 2 * 2      # 2 天 × 三餐人均 170 × 2 人
    assert b.total_hotels == 400 * 1                    # 仅第1天住店，2 人 = 1 间
    assert b.total_transportation == 200                # 交通沿用模型自报
    assert b.total == b.total_attractions + b.total_hotels + b.total_meals + b.total_transportation


def test_budget_ignores_model_reported_totals():
    """工程重算不信模型自报的分项，只按已选 item 的单价重算。"""
    plan = _two_day_plan()
    plan.budget = Budget(total_attractions=99999, total_hotels=99999,
                         total_meals=99999, total_transportation=200, total=99999)
    b = recompute_grounded_budget(plan, party_total=2)
    assert b.total_attractions == 160
    assert b.total != 99999
