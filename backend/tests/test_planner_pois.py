from app.models.schemas import BudgetConstraint, TripRequest
from app.planner import pois


def _req(free_text=""):
    # pois.py 的多个函数无条件访问 request.budget_constraint.budget_level
    # （对齐 helloagents 的 TripRequest，那里 budget_constraint 是必填字段）；
    # 我们的 schema 里 budget_constraint 默认是 None（backend/app/planner/context.py
    # 已经对 None 做了兜底），因此这里显式传入，反映 pois.py 真实依赖的契约。
    return TripRequest(city="成都", start_date="2026-08-01", end_date="2026-08-03",
                       travel_days=3, transportation="打车", accommodation="经济型酒店",
                       preferences=["美食"], free_text_input=free_text,
                       budget_constraint=BudgetConstraint())


def test_parse_location_structured():
    assert pois.parse_location("104.06,30.65") == {"longitude": 104.06, "latitude": 30.65}
    assert pois.parse_location("") is None


def test_food_keyword_groups_has_breakfast_bucket():
    groups = pois.build_food_keyword_groups(_req())
    buckets = {g["bucket"] for g in groups}
    assert "food_breakfast" in buckets  # 早餐单独搜


def test_filter_food_by_constraints_drops_avoided():
    # "不吃辣" -> 应过滤掉辣味候选
    req = _req(free_text="不吃辣")
    rows = [{"name": "麻辣香锅", "type": "川菜;辣"}, {"name": "清粥小菜", "type": "粤菜"}]
    kept = pois.filter_food_by_constraints(rows, req)
    names = {r["name"] for r in kept}
    assert "清粥小菜" in names
