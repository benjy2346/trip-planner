"""TripPlan 输出侧硬校验与预算工程重算。

线上：违规只告警不拦截；训练数据与评测：违规即 hardpass 失败。

两条路径并存到 Task 8：
- `validate_grounded_trip_plan` / `recompute_grounded_budget`：对齐 helloagents
  `eval_rule_metrics.evaluate_output` 语义（四桶景点池、模糊别名 grounding、
  酒店早餐合法、餐厅去重上限、rooms=ceil(人数/2) 预算），新代码一律用它。
- `validate_trip_plan` / `recompute_budget`：旧 hollow-context 版本，仅供 supervisor
  与旧测试在切换前继续可跑，Task 8 删除。
"""
import math
import re
from collections import Counter

from app.models.schemas import TripPlan, Budget
from app.planner.output import (
    is_invalid_hotel_name,
    is_lodging_breakfast_meal,
    is_placeholder_hotel_distance,
    is_placeholder_meal_name,
    meal_diversity_key,
    name_in_candidates,
)

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


# ---------------------------------------------------------------------------
# Grounded 路径（对齐 helloagents eval_rule_metrics.evaluate_output）
# ---------------------------------------------------------------------------

# 景点候选池 = 四桶并集；餐饮候选 = food_pois。
_ATTRACTION_BUCKETS = ("classic_pois", "preference_pois", "scenic_pois", "experience_pois")


def _candidate_names(snapshot: dict, buckets) -> list[str]:
    names: list[str] = []
    for bucket in buckets:
        for item in snapshot.get(bucket) or []:
            if item.get("name"):
                names.append(item["name"])
    return names


def _missing_coord(location) -> bool:
    """坐标为 None 或 (0,0) 占位——旧 hollow context 93% 如此。"""
    if location is None:
        return True
    return location.longitude == 0 and location.latitude == 0


def _meal_repeat_cap(days_count: int) -> int:
    """整趟同一家餐厅最多出现次数：3 天≤2、4-5 天≤3。"""
    return max(2, (days_count * 3 + 4) // 5)


def validate_grounded_trip_plan(plan: TripPlan, context: dict) -> list[str]:
    """对 grounded PlannerContext 产出的 TripPlan 做硬校验。

    读取 helloagents 键名：景点四桶、hotel_pois、food_pois、trip_weather、
    planner_constraints.expected_dates。违规返回中文描述列表（空列表=通过）。
    """
    v: list[str] = []
    req = context["request"]
    snapshot = context["tool_snapshot"]
    dates = context["planner_constraints"]["expected_dates"]
    diet_avoid = [d for d in context.get("preference_profile", {}).get("diet_avoid", []) if d]

    if plan.city != req["city"]:
        v.append(f"city 不一致: {plan.city} != {req['city']}")
    if plan.start_date != req["start_date"] or plan.end_date != req["end_date"]:
        v.append("start_date/end_date 与请求不一致")
    if len(plan.days) != len(dates):
        v.append(f"days 数量 {len(plan.days)} != {len(dates)}")

    attraction_candidates = _candidate_names(snapshot, _ATTRACTION_BUCKETS)
    hotel_candidates = _candidate_names(snapshot, ("hotel_pois",))
    food_candidates = _candidate_names(snapshot, ("food_pois",))
    food_candidate_keys = {k for n in food_candidates if (k := meal_diversity_key(n))}

    meal_key_counts: Counter = Counter()
    attraction_key_counts: Counter = Counter()

    for i, d in enumerate(plan.days):
        label = f"第{i + 1}天"
        is_last = i == len(plan.days) - 1

        if i < len(dates) and d.date != dates[i]:
            v.append(f"{label} date {d.date} != {dates[i]}")
        if d.day_index != i:
            v.append(f"{label} day_index {d.day_index} != {i}")
        if not 1 <= len(d.attractions) <= 3:
            v.append(f"{label} 景点数 {len(d.attractions)} 不在 1-3")

        # 景点 grounding + 坐标 + 跨天去重
        for a in d.attractions:
            if attraction_candidates and not name_in_candidates(a.name, attraction_candidates):
                v.append(f"{label} 景点 {a.name} 不在候选中")
            if _missing_coord(a.location):
                v.append(f"{label} 景点 {a.name} 坐标缺失")
            if key := meal_diversity_key(a.name):
                attraction_key_counts[key] += 1

        # 酒店：中间日不可空、末日应空、名称合法、distance 非占位、坐标
        if not is_last and d.hotel is None:
            v.append(f"{label} 为住宿日但 hotel 为空")
        if d.hotel is not None:
            if is_invalid_hotel_name(d.hotel.name):
                v.append(f"{label} 酒店名非法: {d.hotel.name}")
            elif hotel_candidates and not name_in_candidates(d.hotel.name, hotel_candidates):
                v.append(f"{label} 酒店 {d.hotel.name} 不在候选中")
            if is_placeholder_hotel_distance(d.hotel.distance or ""):
                v.append(f"{label} hotel.distance 为编造占位: {d.hotel.distance}")
            elif d.hotel.distance:
                v.append(f"{label} hotel.distance 应为空字符串: {d.hotel.distance}")
            if _missing_coord(d.hotel.location):
                v.append(f"{label} 酒店 {d.hotel.name} 坐标缺失")

        # 三餐完整 + grounding + 饮食约束 + 同日午晚餐去重
        meal_types = [m.type for m in d.meals]
        for t in _REQUIRED_MEALS:
            if t not in meal_types:
                v.append(f"{label} 缺少 {t}")

        day_lunch_dinner: dict[str, str] = {}
        for m in d.meals:
            mtype = (m.type or "").lower()
            lodging_breakfast = is_lodging_breakfast_meal(m.name, mtype)

            if is_placeholder_meal_name(m.name):
                v.append(f"{label} 餐饮占位词: {m.name}")
            elif not lodging_breakfast and food_candidates and not name_in_candidates(m.name, food_candidates):
                v.append(f"{label} 餐饮 {m.name} 不在候选中")
            if any(bad in m.name for bad in diet_avoid):
                v.append(f"{label} 餐饮 {m.name} 违反饮食约束")

            # 住宿早餐不计入去重统计
            key = meal_diversity_key(m.name)
            if key and not lodging_breakfast:
                meal_key_counts[key] += 1
                if mtype in ("lunch", "dinner"):
                    day_lunch_dinner[mtype] = key

        if (len(food_candidate_keys) >= 2
                and day_lunch_dinner.get("lunch")
                and day_lunch_dinner["lunch"] == day_lunch_dinner.get("dinner")):
            v.append(f"{label} 午晚餐重复: {[m.name for m in d.meals if (m.type or '').lower() in ('lunch', 'dinner')][0]}")

    # 跨天重复上限
    if len(food_candidate_keys) >= 3:
        cap = _meal_repeat_cap(len(plan.days))
        for key, count in meal_key_counts.items():
            if count > cap:
                v.append(f"餐厅重复超限: {key} 出现 {count} 次 (上限 {cap})")
    for key, count in attraction_key_counts.items():
        if count > 1:
            v.append(f"景点重复安排: {key} 出现 {count} 次")

    # 天气逐日复制 trip_weather（仅对有预报的日期）
    snapshot_weather = {w["date"]: w for w in snapshot.get("trip_weather", []) if w.get("date")}
    if snapshot_weather:
        plan_weather = {w.date: w for w in plan.weather_info}
        for day in dates:
            src = snapshot_weather.get(day)
            if src and src.get("source") == "amap_forecast":
                if day not in plan_weather:
                    v.append(f"weather_info 缺少 {day} 的天气")
                elif plan_weather[day].day_weather != src.get("day_weather"):
                    v.append(f"{day} 天气未复制 trip_weather: "
                             f"{plan_weather[day].day_weather} != {src.get('day_weather')}")
    return v


def recompute_grounded_budget(plan: TripPlan, party_total: int) -> Budget:
    """工程重算预算，口径对齐移植的 prompt（不信模型自报分项）。

    - 酒店：单间每晚价 × rooms，rooms = ceil(人数/2)
    - 门票：单人票 × 人数
    - 餐饮：人均单餐 × 人数
    - 交通：沿用模型自报
    """
    party_total = max(1, party_total)
    rooms = math.ceil(party_total / 2)
    hotels = sum(d.hotel.estimated_cost for d in plan.days if d.hotel is not None) * rooms
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
