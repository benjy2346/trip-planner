"""PlannerContext：把请求与结构化取数结果编译成模型可见的开卷资料。

训练数据生成、规则评测、线上推理共用此单一来源，保证三处输入协议一致。

本文件目前并存两条路径：
- `PlannerContextBuilder`（grounded，移植自 helloagents）：真坐标 + 餐饮候选 + 真价格 hint
  + 结构化 preference_profile。新代码一律用它。
- `build_planner_context`（legacy）：靠 LLM 解析子图，坐标多为 0,0、无餐饮候选。
  仅为让 supervisor / ml 脚本 / 旧测试在 Task 8 切换前继续可跑，Task 8 会删除。
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from app.models.schemas import TripRequest, WeatherInfo, Hotel, Attraction
from app.planner.pricing_legacy import hotel_price, meal_cost_table, city_tier

from .amap import AmapPlannerClient
from .compact import compact_for_planner as compact_planner_context
from .pois import (
    annotate_food_pois,
    build_budget_upgrade_keywords,
    build_food_budget_supplement_keywords,
    build_food_keyword_groups,
    build_hotel_keyword_groups,
    build_poi_keywords,
    filter_food_by_constraints,
    merge_poi_buckets,
)
from .policy import build_empty_context
from .pricing import with_hotel_cost_hints, with_meal_cost_hints, with_ticket_price_hints
from .prompts import PLANNER_AGENT_PROMPT
from .weather import align_trip_weather, normalize_weather

from ..logging_config import get_logger

logger = get_logger(__name__)

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


# ---------------------------------------------------------------------------
# Grounded 路径（移植自 helloagents backend/app/planner/context.py）
# MVP 裁剪：不搬 high_end_candidates / local_poi_table / debug。
# ---------------------------------------------------------------------------

PLANNER_CONTEXT_POI_LIMIT = int(os.getenv("PLANNER_CONTEXT_POI_LIMIT", "20"))
PLANNER_CONTEXT_CLASSIC_LIMIT = int(os.getenv("PLANNER_CONTEXT_CLASSIC_LIMIT", "12"))
PLANNER_CONTEXT_PREFERENCE_LIMIT = int(os.getenv("PLANNER_CONTEXT_PREFERENCE_LIMIT", "16"))
PLANNER_CONTEXT_HOTEL_LIMIT = int(os.getenv("PLANNER_CONTEXT_HOTEL_LIMIT", "10"))
PLANNER_CONTEXT_FOOD_LIMIT = int(os.getenv("PLANNER_CONTEXT_FOOD_LIMIT", "40"))
PLANNER_CONTEXT_FOOD_GROUP_LIMIT = int(os.getenv("PLANNER_CONTEXT_FOOD_GROUP_LIMIT", "8"))
PLANNER_CONTEXT_FOOD_BREAKFAST_LIMIT = int(os.getenv("PLANNER_CONTEXT_FOOD_BREAKFAST_LIMIT", "8"))
PLANNER_CONTEXT_FOOD_BASE_LIMIT = int(os.getenv("PLANNER_CONTEXT_FOOD_BASE_LIMIT", "20"))
PLANNER_CONTEXT_FOOD_UPGRADE_LIMIT = int(os.getenv("PLANNER_CONTEXT_FOOD_UPGRADE_LIMIT", "20"))
PLANNER_CONTEXT_FOOD_SUPPLEMENT_LIMIT = int(os.getenv("PLANNER_CONTEXT_FOOD_SUPPLEMENT_LIMIT", "20"))
PLANNER_CONTEXT_HOTEL_UPGRADE_LIMIT = int(os.getenv("PLANNER_CONTEXT_HOTEL_UPGRADE_LIMIT", "6"))
PLANNER_CONTEXT_ATTRACTION_UPGRADE_LIMIT = int(os.getenv("PLANNER_CONTEXT_ATTRACTION_UPGRADE_LIMIT", "10"))
PLANNER_CONTEXT_EXPERIENCE_UPGRADE_LIMIT = int(os.getenv("PLANNER_CONTEXT_EXPERIENCE_UPGRADE_LIMIT", "8"))
FOOD_CONTEXT_MIN_BY_BUDGET_LEVEL = {
    "comfortable": 80,
    "premium": 120,
    "luxury": 180,
}


class PlannerContextBuilder:
    """构造最终 Planner 使用的结构化上下文。"""

    def __init__(self, amap_api_key: str):
        self.amap_client = AmapPlannerClient(amap_api_key)

    def collect(self, request: TripRequest) -> Dict[str, Any]:
        """并行获取 Planner 所需的结构化上下文。"""
        context = self.empty_context(request)
        jobs = {
            "attractions": self._collect_attraction_snapshot,
            "weather": self._collect_weather_snapshot,
            "hotels": self._collect_hotel_snapshot,
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(func, request): name for name, func in jobs.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    context["tool_snapshot"].update(result.get("tool_snapshot", {}))
                    context["tool_snapshot"]["tool_status"][name] = result.get(
                        "status",
                        self._tool_status(True, "ok"),
                    )
                except Exception as exc:
                    context["tool_snapshot"]["tool_status"][name] = self._tool_status(False, str(exc))
                    logger.warning("%s工具快照获取失败: %s", name, exc)

        # 当前主线不生成粗糙路线 hint：直接把酒店/景点/餐饮候选的
        # address、district 和 location 交给 Planner 判断动线。
        context["tool_snapshot"]["route_hints"] = []
        context["tool_snapshot"]["tool_status"]["routes"] = self._tool_status(
            True,
            "disabled; planner uses poi address/district/location",
        )
        return context

    def empty_context(self, request: TripRequest) -> Dict[str, Any]:
        """创建空 PlannerContext，工具失败时也保持输入协议稳定。"""
        return build_empty_context(request)

    def compact_for_planner(self, planner_context: Dict[str, Any]) -> Dict[str, Any]:
        """把原始工具快照压缩成模型真正需要看的 Planner 输入。

        原始 planner_context 仍用于日志和后置校验；这里单独裁剪给 LLM 的版本，
        避免 POI id/typecode/photo_count 等字段把上下文撑得过长。
        """
        return compact_planner_context(planner_context)

    def _collect_attraction_snapshot(self, request: TripRequest) -> Dict[str, Any]:
        """搜索景点、体验和餐饮候选。"""
        classic_keywords = build_poi_keywords(request, "classic")
        preference_keywords = build_poi_keywords(request, "preference")
        experience_keywords = build_poi_keywords(request, "experience")
        food_keyword_groups = build_food_keyword_groups(request)

        classic_pois = self.amap_client.search_classic_pois(
            request.city,
            classic_keywords,
            PLANNER_CONTEXT_CLASSIC_LIMIT,
        )
        if not classic_pois:
            classic_pois = self.amap_client.search_keywords(
                request.city,
                classic_keywords,
                source_role="scenic",
                limit=PLANNER_CONTEXT_CLASSIC_LIMIT,
                source_bucket="classic",
            )
        classic_pois = with_ticket_price_hints(classic_pois, request)

        preference_pois = self.amap_client.search_keywords(
            request.city,
            preference_keywords,
            source_role="scenic",
            limit=PLANNER_CONTEXT_PREFERENCE_LIMIT,
            source_bucket="preference",
        )
        preference_pois = with_ticket_price_hints(preference_pois, request)

        attraction_upgrade_pois = []
        attraction_upgrade_keywords = build_budget_upgrade_keywords(request, "scenic")
        if attraction_upgrade_keywords:
            attraction_upgrade_pois = self.amap_client.search_keywords(
                request.city,
                attraction_upgrade_keywords,
                source_role="scenic",
                limit=PLANNER_CONTEXT_ATTRACTION_UPGRADE_LIMIT,
                source_bucket="attraction_budget_upgrade",
            )
            attraction_upgrade_pois = with_ticket_price_hints(attraction_upgrade_pois, request)

        scenic_pois = merge_poi_buckets(
            [attraction_upgrade_pois, classic_pois, preference_pois],
            PLANNER_CONTEXT_POI_LIMIT,
        )

        experience_pois = self.amap_client.search_keywords(
            request.city,
            experience_keywords,
            source_role="experience",
            limit=PLANNER_CONTEXT_POI_LIMIT,
            source_bucket="experience",
        )
        experience_upgrade_keywords = build_budget_upgrade_keywords(request, "experience")
        if experience_upgrade_keywords:
            experience_upgrade_pois = self.amap_client.search_keywords(
                request.city,
                experience_upgrade_keywords,
                source_role="experience",
                limit=PLANNER_CONTEXT_EXPERIENCE_UPGRADE_LIMIT,
                source_bucket="experience_budget_upgrade",
            )
            experience_pois = merge_poi_buckets(
                [experience_upgrade_pois, experience_pois],
                PLANNER_CONTEXT_POI_LIMIT,
            )
        experience_pois = with_ticket_price_hints(experience_pois, request)

        food_buckets = []
        for group in food_keyword_groups:
            if group["bucket"] == "food_base":
                food_limit = PLANNER_CONTEXT_FOOD_BASE_LIMIT
            elif group["bucket"] == "food_breakfast":
                food_limit = PLANNER_CONTEXT_FOOD_BREAKFAST_LIMIT
            elif group["bucket"] == "food_budget_upgrade":
                food_limit = PLANNER_CONTEXT_FOOD_UPGRADE_LIMIT
            else:
                food_limit = PLANNER_CONTEXT_FOOD_GROUP_LIMIT
            food_buckets.append(
                self.amap_client.search_keywords(
                    request.city,
                    group["keywords"],
                    source_role="food",
                    limit=food_limit,
                    require_location=False,
                    source_bucket=group["bucket"],
                )
            )
        food_pois = merge_poi_buckets(food_buckets, PLANNER_CONTEXT_FOOD_LIMIT)
        food_pois = filter_food_by_constraints(food_pois, request)
        food_pois = annotate_food_pois(food_pois, request)
        food_pois = with_meal_cost_hints(food_pois, request)
        food_pois = self._supplement_food_budget_candidates_if_needed(request, food_pois)

        return {
            "tool_snapshot": {
                "classic_pois": classic_pois,
                "preference_pois": preference_pois,
                "scenic_pois": scenic_pois,
                "experience_pois": experience_pois,
                "food_pois": food_pois,
                "food_query_groups": food_keyword_groups,
            },
            "status": self._tool_status(
                bool(classic_pois or preference_pois or experience_pois),
                (
                    f"classic={len(classic_pois)}, preference={len(preference_pois)}, "
                    f"scenic={len(scenic_pois)}, experience={len(experience_pois)}, "
                    f"food={len(food_pois)}"
                ),
            ),
        }

    def _collect_hotel_snapshot(self, request: TripRequest) -> Dict[str, Any]:
        """搜索酒店候选。"""
        hotel_buckets = []
        for group in build_hotel_keyword_groups(request):
            hotel_limit = (
                PLANNER_CONTEXT_HOTEL_UPGRADE_LIMIT
                if group["bucket"] == "hotel_budget_upgrade"
                else PLANNER_CONTEXT_HOTEL_LIMIT
            )
            hotel_buckets.append(
                self.amap_client.search_keywords(
                    request.city,
                    group["keywords"],
                    source_role="hotel",
                    limit=hotel_limit,
                    require_location=False,
                    source_bucket=group["bucket"],
                )
            )
        hotels = merge_poi_buckets(hotel_buckets, PLANNER_CONTEXT_HOTEL_LIMIT)
        hotels = with_hotel_cost_hints(hotels, request)

        return {
            "tool_snapshot": {"hotel_pois": hotels},
            "status": self._tool_status(bool(hotels), f"hotels={len(hotels)}"),
        }

    def _collect_weather_snapshot(self, request: TripRequest) -> Dict[str, Any]:
        """查询天气，并把短期预报对齐到真实行程日期。"""
        raw = self.amap_client.get("/weather/weatherInfo", {"city": request.city, "extensions": "all"})
        available_weather = normalize_weather(raw)
        trip_weather = align_trip_weather(request, available_weather)
        covered = [item for item in trip_weather if item.get("source") == "amap_forecast"]

        return {
            "tool_snapshot": {
                "available_weather": available_weather,
                "trip_weather": trip_weather,
            },
            "status": self._tool_status(
                bool(available_weather),
                f"available={len(available_weather)}, covered_trip_days={len(covered)}/{request.travel_days}",
            ),
        }

    def _supplement_food_budget_candidates_if_needed(
        self,
        request: TripRequest,
        food_pois: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """高预算午晚餐候选不足时，用更宽的正餐关键词二次补召回。"""
        budget_level = request.budget_constraint.budget_level or ""
        min_cost = FOOD_CONTEXT_MIN_BY_BUDGET_LEVEL.get(budget_level)
        if min_cost is None:
            return food_pois

        target_count = max(request.travel_days * 2, 6)
        if self._count_budget_meal_candidates(food_pois, min_cost) >= target_count:
            return food_pois

        keywords = build_food_budget_supplement_keywords(request)
        if not keywords:
            return food_pois

        supplemental = self.amap_client.search_keywords(
            request.city,
            keywords,
            source_role="food",
            limit=PLANNER_CONTEXT_FOOD_SUPPLEMENT_LIMIT,
            require_location=False,
            source_bucket="food_budget_upgrade",
        )
        supplemental = filter_food_by_constraints(supplemental, request)
        supplemental = annotate_food_pois(supplemental, request)
        supplemental = with_meal_cost_hints(supplemental, request)
        return merge_poi_buckets(
            [food_pois, supplemental],
            max(PLANNER_CONTEXT_FOOD_LIMIT, len(food_pois) + len(supplemental)),
        )

    def _count_budget_meal_candidates(self, rows: List[Dict[str, Any]], min_cost: int) -> int:
        """统计满足当前预算档的 lunch/dinner 候选数。"""
        count = 0
        for row in rows:
            if row.get("source_bucket") == "food_breakfast":
                continue
            roles = {str(role or "").strip().lower() for role in row.get("meal_roles") or []}
            if roles == {"breakfast"}:
                continue
            try:
                cost = int(float(row.get("meal_cost_hint") or 0))
            except (TypeError, ValueError):
                cost = 0
            if cost >= min_cost:
                count += 1
        return count

    def _tool_status(self, ok: bool, message: str) -> Dict[str, Any]:
        """统一工具状态格式，方便 Planner 和日志同时读取。"""
        return {"ok": ok, "message": message}


def build_grounded_planner_messages(context: dict) -> list[BaseMessage]:
    """训练数据生成、规则评测、线上推理共用的 grounded 输入。"""
    return [
        SystemMessage(content=PLANNER_AGENT_PROMPT),
        HumanMessage(content="PlannerContext:\n" + json.dumps(context, ensure_ascii=False)),
    ]
