"""PlannerContext：把请求与子图结构化结果编译成模型可见的开卷资料。

训练数据生成、规则评测、线上推理共用此单一来源，保证三处输入协议一致。
"""
import json
from datetime import date, timedelta
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from app.models.schemas import TripRequest, WeatherInfo, Hotel, Attraction
from app.planner.pricing_legacy import hotel_price, meal_cost_table, city_tier

PRICING_POLICY = {
    "hotel_price_unit": "单间每晚(元)",
    "ticket_price_unit": "成人单人票(元)",
    "meal_cost_unit": "单人单餐(元)",
}

PLANNER_SYSTEM_PROMPT = """你是行程规划专家。输入是一份 JSON 格式的 PlannerContext，包含用户请求、同行人、预算约束、住宿政策、价格口径、工具候选快照和输出约束。你必须只依据这份上下文生成行程，不得编造上下文之外的事实。

硬性规则：
1. 只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字。
2. days 的数量、date、day_index 必须与 planner_constraints.dates 完全一致。
3. weather_info 必须逐日复制 tool_snapshot.weather 的数据，温度为纯数字，不得编造。
4. 每天安排 1-3 个景点，景点必须从 tool_snapshot.attraction_candidates 中选取，并复制其 name/address/location/ticket_price。
5. 除最后一天外每天 hotel 不能为 null，整个行程连续入住同一家酒店（从 tool_snapshot.hotel_candidates 中选取）；最后一天 hotel 为 null。hotel.estimated_cost 必须复制所选候选酒店的 estimated_cost，不得自行编造房价。
6. hotel.distance 必须为空字符串 ""，没有路线工具时不得编造距离。
7. 每天必须包含 breakfast/lunch/dinner 三餐（最后一天也不能缺晚餐），餐饮必须写具体店名，禁止"早餐推荐""附近餐厅""当地小吃""酒店晚餐""无"这类占位词。每餐的 estimated_cost 按 pricing_policy.meal_cost_standard 中对应餐型的标准值填写，不得自行编造。
8. 价格口径见 pricing_policy：酒店按单间每晚（复制候选 estimated_cost），门票按成人单人票（复制候选 ticket_price），餐饮按单人单餐（用 meal_cost_standard）。budget 分项 = 单价 × 对应数量（门票和餐饮要乘 party.total 人数，酒店乘住宿晚数），total 为各分项之和。若预算紧张，优先选更便宜的候选酒店、减少付费景点，把 total 压到 budget_constraint.amount 之内。
9. 若 budget_constraint.strictness 为 "hard"，budget.total 不得超过 budget_constraint.amount。

输出 JSON 结构（与后端 TripPlan schema 一致）：
{
  "city": "...", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD",
  "days": [{
    "date": "YYYY-MM-DD", "day_index": 0, "description": "...",
    "transportation": "...", "accommodation": "...",
    "hotel": {"name": "...", "address": "...", "location": {"longitude": 0.0, "latitude": 0.0},
              "price_range": "...", "rating": "...", "distance": "", "type": "...", "estimated_cost": 0},
    "attractions": [{"name": "...", "address": "...", "location": {"longitude": 0.0, "latitude": 0.0},
                     "visit_duration": 120, "description": "...", "category": "...", "ticket_price": 0}],
    "meals": [{"type": "breakfast", "name": "具体店名", "description": "...", "estimated_cost": 0}]
  }],
  "weather_info": [{"date": "YYYY-MM-DD", "day_weather": "...", "night_weather": "...",
                    "day_temp": 0, "night_temp": 0, "wind_direction": "...", "wind_power": "..."}],
  "overall_suggestions": "...",
  "budget": {"total_attractions": 0, "total_hotels": 0, "total_meals": 0,
             "total_transportation": 0, "total": 0}
}"""


def _dates(request: TripRequest) -> list[str]:
    d = date.fromisoformat(request.start_date)
    return [(d + timedelta(days=i)).isoformat() for i in range(request.travel_days)]


def build_planner_context(
    request: TripRequest,
    weather_outputs: list[WeatherInfo],
    hotel_outputs: list[Hotel],
    poi_outputs: list[Attraction],
) -> dict:
    budget = request.budget_constraint.model_dump() if request.budget_constraint else {
        "amount": None, "scope": "total", "currency": "CNY",
        "budget_level": "comfortable", "strictness": "soft",
    }
    weather = [w.model_dump() for w in weather_outputs]
    hotels = [h.model_dump() for h in hotel_outputs]
    attractions = [p.model_dump() for p in poi_outputs]
    # 给候选酒店贴合成价签（仅当地图侧无价时），作为预算成本信号供模型照抄。
    for idx, h in enumerate(hotels):
        if not h.get("estimated_cost"):
            h["estimated_cost"] = hotel_price(request.accommodation, request.city, idx)
    pricing_policy = dict(PRICING_POLICY)
    pricing_policy["meal_cost_standard"] = meal_cost_table(request.city)
    pricing_policy["city_tier"] = city_tier(request.city)
    return {
        "request": {
            "city": request.city,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "travel_days": request.travel_days,
            "transportation": request.transportation,
            "accommodation": request.accommodation,
            "preferences": request.preferences,
            "free_text_input": request.free_text_input or "",
        },
        "party": request.party.model_dump(),
        "budget_constraint": budget,
        "lodging_policy": {
            "nights": max(request.travel_days - 1, 0),
            "hotel_on_last_day": False,
            "same_hotel_all_nights": True,
        },
        "pricing_policy": pricing_policy,
        "tool_snapshot": {
            "weather": weather,
            "hotel_candidates": hotels,
            "attraction_candidates": attractions,
            "candidate_counts": {
                "weather": len(weather),
                "hotels": len(hotels),
                "attractions": len(attractions),
            },
        },
        "planner_constraints": {
            "days": request.travel_days,
            "dates": _dates(request),
            "attractions_per_day": [1, 3],
            "meals_per_day": ["breakfast", "lunch", "dinner"],
        },
    }


def build_planner_messages(context: dict) -> list[BaseMessage]:
    return [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content="PlannerContext:\n" + json.dumps(context, ensure_ascii=False)),
    ]
