from app.models.schemas import BudgetConstraint, TripRequest
from app.planner import pricing


def _req():
    return TripRequest(city="杭州", start_date="2026-08-01", end_date="2026-08-03",
                       travel_days=3, transportation="打车", accommodation="经济型酒店",
                       budget_constraint=BudgetConstraint())


def test_ticket_price_hint_from_table_or_estimate():
    rows = [{"name": "西湖", "type": "风景名胜"}]
    out = pricing.with_ticket_price_hints(rows, _req())
    assert "ticket_price_hint" in out[0]
    assert isinstance(out[0]["ticket_price_hint"], int)  # 有值（0 也是合法免票），字段存在且为 int


def test_meal_cost_hint_present_and_positive():
    rows = [{"name": "外婆家", "type": "餐饮;浙菜"}]
    out = pricing.with_meal_cost_hints(rows, _req())
    assert out[0].get("meal_cost_hint", 0) > 0


def test_price_table_loads():
    table = pricing.load_attraction_price_table()
    assert isinstance(table, list) and len(table) > 0
