"""Grounded PlannerContext（Task 6）。

断言 Builder 产出的是真 grounding：真坐标、有餐饮候选、有结构化 preference，
且训练/线上共用同一份 prompt + context（train/serve 同源）。
键名一律用 helloagents 词汇表（scenic_pois / food_pois / hotel_pois / trip_weather）。
"""
from unittest.mock import patch

from app.models.schemas import TripRequest
from app.planner.context import PlannerContextBuilder, build_grounded_planner_messages
from app.planner.prompts import PLANNER_AGENT_PROMPT


def _req():
    return TripRequest(
        city="杭州", start_date="2026-08-01", end_date="2026-08-03", travel_days=3,
        transportation="打车", accommodation="经济型酒店", preferences=["美食"],
        free_text_input="不吃辣", party={"adults": 2},
        budget_constraint={"amount": 4000, "strictness": "hard"},
    )


def _rows(kind, n=4):
    return [
        {
            "name": f"{kind}{i}",
            "address": "西湖区某路1号",
            "adname": "西湖区",
            "location": {"longitude": 120.1 + i / 100, "latitude": 30.2 + i / 100},
            "type": "餐饮服务;中餐厅" if kind == "food" else "风景名胜;公园广场",
            "meal_cost_hint": 60 if kind == "food" else None,
            "ticket_price_hint": 40 if kind == "poi" else None,
            "estimated_cost_hint": 400 if kind == "hotel" else None,
        }
        for i in range(n)
    ]


def _build():
    """Builder + patch 掉三个快照方法（不打真高德）。"""
    b = PlannerContextBuilder(amap_api_key="TESTKEY")
    attractions = {
        "tool_snapshot": {
            "classic_pois": _rows("poi"),
            "preference_pois": _rows("poi"),
            "scenic_pois": _rows("poi"),
            "experience_pois": [],
            "food_pois": _rows("food"),
            "food_query_groups": [{"bucket": "food_base", "keywords": ["杭帮菜"]}],
        },
        "status": {"ok": True, "message": "stub"},
    }
    hotels = {"tool_snapshot": {"hotel_pois": _rows("hotel")}, "status": {"ok": True, "message": "stub"}}
    weather = {"tool_snapshot": {"available_weather": [], "trip_weather": []},
               "status": {"ok": True, "message": "stub"}}
    return b, attractions, hotels, weather


def _collect():
    b, attractions, hotels, weather = _build()
    with patch.object(b, "_collect_attraction_snapshot", return_value=attractions), \
         patch.object(b, "_collect_hotel_snapshot", return_value=hotels), \
         patch.object(b, "_collect_weather_snapshot", return_value=weather):
        return b, b.collect(_req())


def test_collect_produces_grounded_candidates():
    _, ctx = _collect()
    snap = ctx["tool_snapshot"]
    for key in ("scenic_pois", "hotel_pois", "food_pois"):
        assert snap[key], f"{key} 不应为空"


def test_candidates_carry_real_coordinates():
    """旧 hollow context 93% 景点坐标是 0,0——这里必须是真经纬度。"""
    _, ctx = _collect()
    snap = ctx["tool_snapshot"]
    for key in ("scenic_pois", "hotel_pois", "food_pois"):
        for row in snap[key]:
            loc = row.get("location")
            assert loc is not None, f"{key} 候选缺少 location"
            assert (loc["longitude"], loc["latitude"]) != (0, 0), f"{key} 候选坐标为 0,0 占位"


def test_context_carries_structured_preference_profile():
    _, ctx = _collect()
    profile = ctx["preference_profile"]
    assert "辣" in profile["diet_avoid"], "free_text_input『不吃辣』应进 diet_avoid"
    assert profile["positive_preferences"]


def test_compact_keeps_food_and_preference():
    b, ctx = _collect()
    compact = b.compact_for_planner(ctx)
    assert compact["tool_snapshot"]["food_pois"], "compact 后仍需保留餐饮候选"
    assert compact["preference_profile"]["diet_avoid"]


def test_messages_are_train_serve_identical():
    """训练数据生成与线上推理必须用同一 prompt + 同一 context 序列化。"""
    b, ctx = _collect()
    compact = b.compact_for_planner(ctx)
    msgs = build_grounded_planner_messages(compact)
    assert len(msgs) == 2
    assert msgs[0].content == PLANNER_AGENT_PROMPT
    assert "food0" in msgs[1].content, "餐饮候选应出现在模型可见的 context 里"
