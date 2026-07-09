"""TripPlan 输出侧硬校验与预算工程重算。

线上：违规只告警不拦截；训练数据与评测：违规即 hardpass 失败。
"""
import re
from app.models.schemas import TripPlan, Budget

MEAL_PLACEHOLDER_RE = re.compile(
    r"(推荐|附近餐厅|附近小吃|当地美食|当地小吃|特色小吃|酒店早餐|酒店午餐|酒店晚餐|自理|待定)|^无"
)
_REQUIRED_MEALS = ("breakfast", "lunch", "dinner")


def validate_trip_plan(plan: TripPlan, context: dict) -> list[str]:
    v: list[str] = []
    req = context["request"]
    dates = context["planner_constraints"]["dates"]
    snapshot = context["tool_snapshot"]

    if plan.city != req["city"]:
        v.append(f"city 不一致: {plan.city} != {req['city']}")
    if plan.start_date != req["start_date"] or plan.end_date != req["end_date"]:
        v.append("start_date/end_date 与请求不一致")
    if len(plan.days) != len(dates):
        v.append(f"days 数量 {len(plan.days)} != {len(dates)}")

    hotel_names = {h["name"] for h in snapshot["hotel_candidates"]}
    attraction_names = {a["name"] for a in snapshot["attraction_candidates"]}

    for i, d in enumerate(plan.days):
        label = f"第{i + 1}天"
        if i < len(dates) and d.date != dates[i]:
            v.append(f"{label} date {d.date} != {dates[i]}")
        if d.day_index != i:
            v.append(f"{label} day_index {d.day_index} != {i}")
        if not 1 <= len(d.attractions) <= 3:
            v.append(f"{label} 景点数 {len(d.attractions)} 不在 1-3")

        meal_types = [m.type for m in d.meals]
        for t in _REQUIRED_MEALS:
            if t not in meal_types:
                v.append(f"{label} 缺少 {t}")
        for m in d.meals:
            if MEAL_PLACEHOLDER_RE.search(m.name):
                v.append(f"{label} 餐饮占位词: {m.name}")

        is_last = i == len(plan.days) - 1
        if not is_last and d.hotel is None:
            v.append(f"{label} 为住宿日但 hotel 为空")
        if d.hotel is not None:
            if d.hotel.distance:
                v.append(f"{label} hotel.distance 应为空字符串: {d.hotel.distance}")
            if hotel_names and d.hotel.name not in hotel_names:
                v.append(f"{label} 酒店 {d.hotel.name} 不在候选中")
        if attraction_names:
            for a in d.attractions:
                if a.name not in attraction_names:
                    v.append(f"{label} 景点 {a.name} 不在候选中")

    snapshot_weather = {w["date"]: w for w in snapshot["weather"]}
    if snapshot_weather:
        plan_weather = {w.date: w for w in plan.weather_info}
        for day in dates:
            if day in snapshot_weather:
                if day not in plan_weather:
                    v.append(f"weather_info 缺少 {day} 的天气")
                elif plan_weather[day].day_weather != snapshot_weather[day]["day_weather"]:
                    v.append(f"{day} 天气未复制 tool_snapshot: "
                             f"{plan_weather[day].day_weather} != {snapshot_weather[day]['day_weather']}")
    return v


def recompute_budget(plan: TripPlan, party_total: int) -> Budget:
    """工程重算预算：酒店按晚、门票×人数、餐饮×人数；交通沿用模型自报。"""
    hotels = sum(d.hotel.estimated_cost for d in plan.days if d.hotel is not None)
    attractions = sum(a.ticket_price for d in plan.days for a in d.attractions) * party_total
    meals = sum(m.estimated_cost for d in plan.days for m in d.meals) * party_total
    transportation = plan.budget.total_transportation if plan.budget else 0
    return Budget(
        total_attractions=attractions,
        total_hotels=hotels,
        total_meals=meals,
        total_transportation=transportation,
        total=attractions + hotels + meals + transportation,
    )
