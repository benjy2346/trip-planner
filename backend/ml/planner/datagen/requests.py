"""受控合成 TripRequest 生成器（controlled request source）。

只覆盖 `--request-source controlled` 这一路：按近似真实的城市/日期/同行人数/预算/
饮食/偏好分布加权抽样出结构化 `TripRequest`，并附带一份 `control_spec` 元数据块，
供 Task 5 的 record builder 驱动 teacher 生成 + 事后审计切片。

不覆盖 `generate_template_requests` / `generate_llm_request(s)`（`template`/`llm`
两种替代请求源，2a 只用 controlled，不在本模块范围内）。

行为对齐 helloagents 参考实现
（training/scripts/planner/data/generate_sft_data.py:459-1165 的 controlled 分支），
但结构和函数拆分是我们自己写的，不是逐行照抄；权重表和推导公式保持数值一致，
是造数分布的一部分，必须保真。

⚠️ free_text 只能使用 `app.planner.policy.build_preference_profile` /
`app.planner.pois.infer_food_constraints` 真正识别的关键词——否则 control_spec
里写的约束在真实 preference_profile 里就是"死"的（Task 5 消费的是后者，不是
control_spec 本身）。avoid_long_walk 用 少走路/不想太累/行动不便/无障碍/轮椅，
不用参考实现里对不上号的"少爬山"。
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Any

from app.models.schemas import TripRequest
from app.planner.policy import NEGATIVE_CONSTRAINT_PHRASES
from app.planner.pois import NEGATIVE_PREFERENCE_MARKERS, infer_food_constraints

# ---------------------------------------------------------------------------
# 城市/同行/日期/预算/节奏/饮食 分布权重
# ---------------------------------------------------------------------------

REQUEST_ID_PREFIX = "request"

CITY_TIERS: dict[str, list[str]] = {
    "major": ["北京", "上海", "广州", "深圳", "成都", "杭州", "重庆", "西安"],
    "popular": ["南京", "苏州", "厦门", "青岛", "长沙", "武汉", "昆明", "大理", "丽江", "桂林", "三亚", "哈尔滨"],
    "long_tail": [
        "珠海", "福州", "泉州", "天津", "洛阳", "扬州", "贵阳", "呼和浩特",
        "沈阳", "大连", "宁波", "济南", "郑州", "黄山", "张家界",
    ],
}
CITY_TIER_WEIGHTS: list[tuple[str, int]] = [("major", 35), ("popular", 40), ("long_tail", 25)]

COMPANION_WEIGHTS: list[tuple[str, int]] = [
    ("friends", 22),
    ("couple", 20),
    ("solo", 15),
    ("family_with_children", 18),
    ("family_with_elders", 10),
    ("family_mixed", 7),
    ("business", 4),
    ("other", 4),
]

# choose_controlled_start_date 的 date_mode="mixed" 分支：past/near_future/far_future
# 三档混合，保证历史天气（Task 1 Open-Meteo）真的有过去行程可用。
DATE_MODE_WEIGHTS: list[tuple[str, int]] = [("past", 30), ("near_future", 30), ("far_future", 40)]

TRAVEL_DAYS_WEIGHTS: list[tuple[int, int]] = [(2, 18), (3, 42), (4, 28), (5, 12)]
BUDGET_WEIGHTS: list[tuple[str, int]] = [
    ("limited", 25), ("standard", 40), ("comfortable", 23), ("premium", 9), ("luxury", 3),
]
PACE_WEIGHTS: list[tuple[str, int]] = [
    ("适中节奏", 45), ("慢节奏", 35), ("紧凑高效", 12), ("自由活动多", 8),
]
DIET_WEIGHTS: list[tuple[str, int]] = [
    # "清淡饮食" 从参考实现继承的标签在 app.planner.pois.DIET_PREFERENCE_KEYWORDS 里没有对应词，
    # 永远无法在真实 preference_profile 里出现（dead label），故不采用；权重并入"无"。
    ("无", 77), ("少辣", 10), ("海鲜过敏", 5), ("清真", 4), ("素食", 4),
]

# 整趟总预算口径：住宿按 N-1 晚两人一间、餐饮门票按人数线性、市内交通按队伍共享日成本，
# 城市层级做整体系数微调。必须和 app/planner 侧的价格口径保持同一套语言（元/CNY）。
PER_PERSON_DAY_BUDGETS: dict[str, list[int]] = {
    "limited": [220, 270, 320],
    "standard": [380, 470, 560],
    "comfortable": [600, 750, 900],
    "premium": [950, 1200, 1450],
    "luxury": [1600, 2200, 3000],
}
REQUEST_HOTEL_COST_BY_ACCOMMODATION: dict[str, int] = {
    "经济型酒店": 300, "民宿": 420, "舒适型酒店": 520, "亲子酒店": 650, "高端酒店": 1000,
}
REQUEST_CITY_BUDGET_FACTORS: dict[str, float] = {"major": 1.15, "popular": 1.05, "long_tail": 0.95}
REQUEST_SHARED_TRANSPORT_DAY_COST: dict[str, int] = {
    "公共交通": 80, "地铁+步行": 60, "打车": 220, "自驾": 260,
}

THEME_POOL: list[tuple[str, int]] = [
    ("美食", 38), ("历史文化", 28), ("博物馆", 18), ("自然风光", 25), ("城市公园", 16),
    ("休闲慢游", 30), ("城市地标", 22), ("第一次来", 18), ("摄影", 15), ("城市漫步", 15),
    ("购物商圈", 10), ("主题乐园", 10), ("户外轻徒步", 8), ("夜市夜景", 8),
    ("小众展览", 7), ("艺术", 7), ("海滨度假", 6),
]

# diet_positive / diet_avoid 不再用手写的 label->词表映射（曾经是死数据源：声明的忌口和
# app.planner.pois.infer_food_constraints 从 free_text 实际抽出的忌口对不上）。改为在
# generate_controlled_request 里对已生成的 TripRequest 直接调用 infer_food_constraints，
# 保证 control_spec.diet_avoid/diet_positive 和模型实际看到的 preference_profile 一致。

# NEGATIVE_PREFERENCE_MARKERS 直接复用 app.planner.pois 的定义（上面已 import），
# 不在本模块重复声明，避免和真实识别词表脱节（曾经手写的 9 项拷贝漏了"禁忌"/"少走路"）。

# avoid_long_walk 只能用 app.planner.policy.build_preference_profile 实际识别的词，
# 否则 control_spec 里的 avoid_long_walk=True 在真实 preference_profile 里永远是 False。
AVOID_LONG_WALK_MARKERS = ["少走路", "不想太累", "行动不便", "无障碍", "轮椅"]

# 住宿方案：按预算档位分布，family_with_children / family_mixed 额外偏向亲子酒店。
_ACCOMMODATION_WEIGHTS_BY_BUDGET: dict[str, list[tuple[str, int]]] = {
    "luxury": [("高端酒店", 66), ("舒适型酒店", 24), ("民宿", 10)],
    "premium": [("高端酒店", 45), ("舒适型酒店", 38), ("民宿", 10), ("经济型酒店", 7)],
    "comfortable": [("舒适型酒店", 55), ("高端酒店", 18), ("民宿", 17), ("经济型酒店", 10)],
    "limited": [("经济型酒店", 70), ("民宿", 20), ("舒适型酒店", 10)],
    "standard": [("舒适型酒店", 48), ("经济型酒店", 35), ("民宿", 16), ("高端酒店", 1)],
}
_ACCOMMODATION_WEIGHTS_BY_COMPANION: dict[str, dict[str, list[tuple[str, int]]]] = {
    "family_with_children": {
        "limited": [("经济型酒店", 50), ("舒适型酒店", 28), ("亲子酒店", 14), ("民宿", 8)],
        "standard": [("舒适型酒店", 42), ("亲子酒店", 30), ("经济型酒店", 20), ("民宿", 8)],
        "comfortable": [("舒适型酒店", 45), ("亲子酒店", 32), ("高端酒店", 12), ("民宿", 11)],
        "premium": [("亲子酒店", 36), ("高端酒店", 32), ("舒适型酒店", 24), ("民宿", 8)],
        "luxury": [("亲子酒店", 36), ("高端酒店", 32), ("舒适型酒店", 24), ("民宿", 8)],
    },
    "family_mixed": {
        "limited": [("经济型酒店", 48), ("舒适型酒店", 34), ("民宿", 12), ("亲子酒店", 6)],
        "standard": [("舒适型酒店", 50), ("经济型酒店", 22), ("亲子酒店", 18), ("民宿", 10)],
        "comfortable": [("舒适型酒店", 48), ("亲子酒店", 26), ("高端酒店", 14), ("民宿", 12)],
        "premium": [("高端酒店", 36), ("舒适型酒店", 32), ("亲子酒店", 24), ("民宿", 8)],
        "luxury": [("高端酒店", 36), ("舒适型酒店", 32), ("亲子酒店", 24), ("民宿", 8)],
    },
}

_METRO_CITIES = {"北京", "上海", "广州", "深圳", "成都", "杭州", "重庆", "南京", "苏州", "武汉", "西安"}
_SCENIC_TRANSPORT_CITIES = {"大理", "丽江", "桂林", "三亚", "张家界", "黄山"}


# ---------------------------------------------------------------------------
# 通用加权抽样
# ---------------------------------------------------------------------------

def weighted_choice(rng: random.Random, weighted_items: list[tuple[Any, int]]) -> Any:
    """按整数权重抽样单个值。"""
    total = sum(weight for _, weight in weighted_items)
    point = rng.uniform(0, total)
    upto = 0.0
    for item, weight in weighted_items:
        upto += weight
        if point <= upto:
            return item
    return weighted_items[-1][0]


def weighted_block_choice(index: int, seed: int, weighted_items: list[tuple[Any, int]], salt: str) -> Any:
    """按权重构造一个确定性排列块，同一 (seed, salt) 下按 index 取值可复现。"""
    pool: list[Any] = []
    for item, weight in weighted_items:
        pool.extend([item] * weight)
    if not pool:
        raise ValueError("weighted_items 不能为空")
    rng = random.Random(f"{seed}:{salt}:{len(pool)}")
    rng.shuffle(pool)
    return pool[index % len(pool)]


def sample_many_weighted(
    rng: random.Random, weighted_items: list[tuple[str, int]], min_count: int, max_count: int,
) -> list[str]:
    """不放回地按权重抽多个主题标签。"""
    candidates = list(weighted_items)
    results: list[str] = []
    target = rng.randint(min_count, max_count)
    while candidates and len(results) < target:
        item = weighted_choice(rng, candidates)
        results.append(item)
        candidates = [(name, weight) for name, weight in candidates if name != item]
    return results


def unique_list(values: list[Any]) -> list[str]:
    """保持顺序去重的字符串列表。"""
    results: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


# ---------------------------------------------------------------------------
# 同行人数 / 城市层级 / 预算金额 / 预算约束
# ---------------------------------------------------------------------------

def build_party_info(rng: random.Random, companion_type: str) -> dict[str, Any]:
    """按同行类型生成结构化人数。"""
    if companion_type == "solo":
        adults, children, elders = 1, 0, 0
    elif companion_type == "couple":
        adults, children, elders = 2, 0, 0
    elif companion_type == "friends":
        adults, children, elders = rng.choice([2, 3, 4]), 0, 0
    elif companion_type == "family_with_children":
        adults, children, elders = 2, rng.choice([1, 1, 2]), 0
    elif companion_type == "family_with_elders":
        adults, children, elders = rng.choice([1, 2]), 0, rng.choice([1, 2])
    elif companion_type == "family_mixed":
        adults, children, elders = 2, rng.choice([1, 1, 2]), rng.choice([1, 2])
    elif companion_type == "business":
        adults, children, elders = rng.choice([1, 2, 3]), 0, 0
    else:
        adults, children, elders = rng.choice([1, 2, 3]), 0, 0

    return {
        "adults": adults,
        "children": children,
        "elders": elders,
        "total": adults + children + elders,
        "companion_type": companion_type,
    }


def infer_city_tier(city: str) -> str:
    """根据城市名回推预算城市层级，未知城市按 popular 处理。"""
    for tier, cities in CITY_TIERS.items():
        if city in cities:
            return tier
    return "popular"


def choose_budget_amount(
    rng: random.Random,
    budget_level: str,
    party_total: int,
    travel_days: int,
    accommodation: str,
    transportation: str = "公共交通",
    city_tier: str = "popular",
) -> int:
    """按整趟总预算口径推导预算金额，百元取整。

    住宿=N-1晚两人一间；餐饮/门票/体验按人数*天数线性；市内交通按队伍共享日成本；
    城市层级做整体系数微调。luxury 档位的人均日预算沿用 premium 档，避免极端离群值
    （对齐参考实现里同样的取舍：luxury 只影响住宿/主题选择，不放大人均预算区间）。
    """
    per_person_level = "premium" if budget_level == "luxury" else budget_level
    per_person_day = rng.choice(PER_PERSON_DAY_BUDGETS.get(per_person_level, PER_PERSON_DAY_BUDGETS["standard"]))

    party_total = max(party_total, 1)
    travel_days = max(travel_days, 1)
    lodging_nights = max(travel_days - 1, 0)
    lodging_per_night = REQUEST_HOTEL_COST_BY_ACCOMMODATION.get(accommodation, 300)
    room_count = max(1, math.ceil(party_total / 2))
    city_factor = REQUEST_CITY_BUDGET_FACTORS.get(city_tier, 1.0)
    shared_transport_day = REQUEST_SHARED_TRANSPORT_DAY_COST.get(transportation, 120)

    lodging_total = lodging_per_night * lodging_nights * room_count
    person_total = per_person_day * party_total * travel_days
    shared_transport_total = shared_transport_day * travel_days
    raw_total = (lodging_total + person_total + shared_transport_total) * city_factor
    return max(500, int(round(raw_total / 100.0) * 100))


def build_budget_constraint(rng: random.Random, budget_level: str, amount: int) -> dict[str, Any]:
    """构造预算约束 dict（budget_level/strictness 值域对齐 app.planner 的价格/偏好关键词表）。

    唯一调用点（generate_controlled_request）里 amount 恒为 choose_budget_amount 算出的整数
    （从不为 None），也从不传入 strictness——原来的 `if strictness / elif amount is None`
    分支永远走不到，这里直接去掉，只保留真正会执行的 budget_level 加权抽样。
    """
    if budget_level == "limited":
        resolved = weighted_choice(rng, [("soft", 72), ("hard", 20), ("none", 8)])
    else:
        resolved = weighted_choice(rng, [("soft", 88), ("hard", 4), ("none", 8)])

    return {
        "amount": amount,
        "scope": "total",
        "currency": "CNY",
        "budget_level": budget_level,
        "strictness": resolved,
    }


# ---------------------------------------------------------------------------
# 日期 / 住宿 / 交通
# ---------------------------------------------------------------------------

def choose_controlled_start_date(rng: random.Random, travel_days: int, date_mode: str) -> date:
    """受控分布下的出发日期抽样。

    date_mode="mixed" 按 DATE_MODE_WEIGHTS 混合 past/near_future/far_future 三档，
    保证一批数据里既有已结束的历史行程（触发 Task 1 的 Open-Meteo 历史天气），
    也有近期/远期行程（触发高德短期天气预报）。
    """
    if date_mode == "past":
        bucket = "past"
    elif date_mode == "future":
        bucket = weighted_choice(rng, [("near_future", 35), ("far_future", 65)])
    else:
        bucket = weighted_choice(rng, DATE_MODE_WEIGHTS)

    if bucket == "past":
        start_delta = -rng.choice([travel_days + 1, 7, 14, 30, 60, 90, 120, 180, 270, 365])
    elif bucket == "near_future":
        start_delta = rng.choice([1, 2, 3, 4])
    else:
        start_delta = rng.choice([30, 45, 60, 90, 120])
    return date.today() + timedelta(days=start_delta)


def choose_controlled_accommodation(rng: random.Random, companion_type: str, budget_level: str) -> str:
    """住宿分布：预算档位为主，亲子/三代同游类同行类型额外偏向亲子酒店。"""
    by_companion = _ACCOMMODATION_WEIGHTS_BY_COMPANION.get(companion_type)
    if by_companion:
        weights = by_companion.get(budget_level, by_companion["standard"])
    else:
        weights = _ACCOMMODATION_WEIGHTS_BY_BUDGET.get(budget_level, _ACCOMMODATION_WEIGHTS_BY_BUDGET["standard"])
    return weighted_choice(rng, weights)


def choose_controlled_transportation(rng: random.Random, companion_type: str, city: str) -> str:
    """交通方式分布：同行类型优先，其次按城市地铁覆盖/景区属性微调。"""
    if companion_type in {"family_with_children", "family_with_elders", "family_mixed"}:
        weights = [("地铁+步行", 26), ("公共交通", 18), ("打车", 42), ("自驾", 14)]
    elif companion_type == "business":
        weights = [("打车", 56), ("地铁+步行", 22), ("公共交通", 12), ("自驾", 10)]
    elif city in _SCENIC_TRANSPORT_CITIES:
        weights = [("地铁+步行", 12), ("公共交通", 22), ("打车", 34), ("自驾", 32)]
    elif city in _METRO_CITIES:
        weights = [("地铁+步行", 50), ("公共交通", 28), ("打车", 18), ("自驾", 4)]
    else:
        weights = [("地铁+步行", 34), ("公共交通", 30), ("打车", 24), ("自驾", 12)]
    return weighted_choice(rng, weights)


# ---------------------------------------------------------------------------
# 自由文本 + control_spec
# ---------------------------------------------------------------------------

def choose_budget_text(amount: int | None) -> str:
    if amount is None:
        return "预算没有特别限制"
    return f"总预算控制在{amount}元左右"


def companion_phrase(rng: random.Random, companion_type: str) -> str:
    """同行类型转自然语气短句。"""
    if companion_type == "family_with_children":
        age = rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
        return rng.choice([f"带{age}岁孩子", f"一家三口带{age}岁小朋友", f"带娃出行，孩子{age}岁"])
    if companion_type == "friends":
        return rng.choice(["和朋友一起", "几个朋友出行", "和同学/朋友一起"])
    if companion_type == "couple":
        return rng.choice(["情侣两个人", "和对象一起", "夫妻两个人"])
    if companion_type == "solo":
        return rng.choice(["一个人出行", "独自旅行", "单人自由行"])
    if companion_type == "family_with_elders":
        return rng.choice(["带父母出行", "陪长辈一起", "和爸妈一起"])
    if companion_type == "family_mixed":
        return rng.choice(["三代同游", "带父母和孩子一起", "一家老小一起出行"])
    if companion_type == "business":
        return rng.choice(["出差顺便玩一天", "商务行程后想轻松逛逛", "和同事短途出行"])
    return rng.choice(["这次想轻松玩几天", "第一次认真规划这个城市", "想安排一个舒服点的短途旅行"])


def build_controlled_free_text(
    rng: random.Random, companion_type: str, budget_amount: int | None, diet: str, pace: str, avoid: list[str],
) -> str:
    """拼出自然语言补充说明，不调用强模型。

    每一句都要用 app.planner.policy/pois 真正识别的关键词，否则 control_spec 里声明
    的约束在下游 preference_profile 里就是死的：
    - diet_text 必须同时命中 pois.DIET_PREFERENCE_KEYWORDS（正向 diet 标签）和/或
      pois.FOOD_AVOID_KEYWORDS + FOOD_AVOID_MARKERS 紧邻搭配（忌口 avoid），逐label验证：
        * 海鲜过敏 -> "海鲜过敏"（keyword+marker 紧邻）-> avoid=[海鲜]；"海鲜"本身在
          DIET_PREFERENCE_KEYWORDS 里但因为已进 avoid 被跳过，diet="无"（这是预期行为：
          过敏是忌口不是正向饮食偏好）。
        * 清真 -> "清真"关键词命中 diet="清真"；"不吃猪肉、忌酒" -> marker+keyword 紧邻
          -> avoid=[猪肉,酒]。
        * 素食 -> "素食"关键词命中 diet="素食"；"不吃牛肉、不吃羊肉、不吃猪肉、不吃海鲜"
          （每个词前都重复"不吃"以满足紧邻搭配）-> avoid=[海鲜,牛肉,羊肉,猪肉]。
        * 少辣 -> 文本同时含"少辣"（diet 关键词）和"不吃辣"（marker+keyword 紧邻）
          -> diet="少辣" 且 avoid=[辣] 同时成立，这是刻意选择的一致处理方式（少辣既是
          正向饮食标签也隐含忌口"辣"），不是冲突。
      "清淡饮食" 不在 DIET_PREFERENCE_KEYWORDS 里，永远无法作为 diet 出现，已从
      DIET_WEIGHTS 里整体去掉（见该常量定义处注释），这里不再处理这个 label。
    - avoid 的每一项都取自 app.planner.policy.NEGATIVE_CONSTRAINT_PHRASES，逐字命中。
    - mobility/无障碍诉求主动补一句用 AVOID_LONG_WALK_MARKERS 里的真词（少走路/不想太累），
      不用参考实现里那种从不触发 avoid_long_walk 的写法（它的 avoid 词池里没有一个词能
      命中 avoid_long_walk 的关键词表，等于这条约束在受控源里永远是死的）。
    """
    parts = [companion_phrase(rng, companion_type), choose_budget_text(budget_amount)]

    if pace == "慢节奏":
        parts.append("希望节奏慢一点，不要每天赶太多景点")
    elif pace == "紧凑高效":
        parts.append("可以紧凑一点，想多看几个经典地方")
    elif pace == "自由活动多":
        parts.append("希望每天留一点自由活动时间")
    else:
        parts.append("节奏适中就行")

    diet_text = {
        "少辣": "口味希望少辣，尽量不吃辣",
        "海鲜过敏": "对海鲜过敏，不要安排海鲜餐厅",
        "清真": "有清真饮食要求，不吃猪肉、忌酒",
        "素食": "素食，不吃牛肉、不吃羊肉、不吃猪肉、不吃海鲜",
    }.get(diet)
    if diet_text:
        parts.append(diet_text)

    if avoid:
        parts.append(f"尽量避开{'、'.join(avoid)}")

    if companion_type in {"family_with_elders", "family_mixed"} and rng.random() < 0.5:
        parts.append("尽量少走路，行程不想太累")

    return "，".join(parts) + "。"


def positive_preference_tags(preferences: list[str]) -> list[str]:
    """造数侧保证 preferences 只承载正向偏好（过滤掉明显是负向表达的项）。"""
    return unique_list(
        [item for item in preferences if not any(marker in str(item) for marker in NEGATIVE_PREFERENCE_MARKERS)]
    )


def build_negative_constraints(avoid_pool: list[str], food_avoid: list[str], free_text: str) -> list[str]:
    """构造造数侧的负向约束标签：显式 avoid_pool 列表 + 真实饮食忌口 + free_text 里命中的负向短语。

    food_avoid 必须是 infer_food_constraints(...)["avoid"] 算出来的真实忌口列表，不是
    手写映射——否则又会退回到"control_spec 声明的忌口和真实 preference_profile 对不上"
    的老问题。
    """
    results = list(avoid_pool) + list(food_avoid)
    for phrase in NEGATIVE_CONSTRAINT_PHRASES:
        if phrase in free_text:
            results.append(phrase)
    return unique_list(results)


def build_preference_control_spec(
    preferences: list[str],
    free_text: str,
    party: dict[str, Any],
    pace: str,
    avoid_pool: list[str],
    food_constraints: dict[str, Any],
) -> dict[str, Any]:
    """把正向偏好/负向约束显式写入 control_spec，键名对齐 app.planner.policy.build_preference_profile。

    diet_positive/diet_avoid 直接取自 food_constraints（由调用方对已生成的 TripRequest
    调用 app.planner.pois.infer_food_constraints 算出），保证和模型实际看到的
    preference_profile 逐字段一致，不能再退化成造数侧自说自话的手写映射。
    """
    traveler_constraints = {
        "needs_child_friendly": int(party.get("children") or 0) > 0 or "孩子" in free_text or "带娃" in free_text,
        "needs_elder_friendly": int(party.get("elders") or 0) > 0 or "老人" in free_text or "长辈" in free_text,
        "avoid_long_walk": any(marker in free_text for marker in AVOID_LONG_WALK_MARKERS),
    }
    diet = food_constraints["diet"]
    food_avoid = food_constraints["avoid"]
    return {
        "positive_preferences": positive_preference_tags(preferences),
        "negative_constraints": build_negative_constraints(avoid_pool, food_avoid, free_text),
        "diet_positive": [] if diet == "无" else [diet],
        "diet_avoid": food_avoid,
        "pace": pace,
        "traveler_constraints": traveler_constraints,
    }


def format_request_id(index: int) -> str:
    return f"{REQUEST_ID_PREFIX}_{index:06d}"


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

_AVOID_POOL = ["人挤人的网红店", "过度商业化景点", "太累的路线", "太偏远的景点", "高价餐厅", "购物团"]


def generate_controlled_request(index: int, *, seed: int, date_mode: str) -> dict[str, Any]:
    """按近似真实分布生成一条 TripRequest 素材 + control_spec，不调用高德/强模型。

    同一 (seed, index) 永远产出同一条记录：内部只用一个 `random.Random(seed + index)`，
    没有任何跨记录共享的可变状态。
    """
    rng = random.Random(seed + index)

    companion_type = weighted_choice(rng, COMPANION_WEIGHTS)
    city_tier = weighted_choice(rng, CITY_TIER_WEIGHTS)
    city = rng.choice(CITY_TIERS[city_tier])
    travel_days = weighted_choice(rng, TRAVEL_DAYS_WEIGHTS)
    budget_level = weighted_choice(rng, BUDGET_WEIGHTS)
    pace = weighted_choice(rng, PACE_WEIGHTS)
    diet = weighted_choice(rng, DIET_WEIGHTS)

    party = build_party_info(rng, companion_type)
    accommodation = choose_controlled_accommodation(rng, companion_type, budget_level)
    transportation = choose_controlled_transportation(rng, companion_type, city)
    budget_amount = choose_budget_amount(
        rng, budget_level, party["total"], travel_days, accommodation, transportation, city_tier,
    )
    budget_constraint = build_budget_constraint(rng, budget_level, budget_amount)
    start = choose_controlled_start_date(rng, travel_days, date_mode)

    themes = sample_many_weighted(rng, THEME_POOL, 2, 4)
    if companion_type == "family_with_children" and "亲子" not in themes:
        themes = ["亲子"] + themes[:3]
    if companion_type == "family_with_elders" and "老人友好" not in themes:
        themes = ["老人友好"] + themes[:3]
    if companion_type == "family_mixed":
        themes = unique_list(["亲子", "老人友好"] + themes)[:4]
    if diet in {"清真", "素食"} and diet not in themes:
        themes = themes[:3] + [diet]

    avoid = rng.sample(_AVOID_POOL, k=rng.choice([1, 1, 2, 2, 3])) if rng.random() < 0.68 else []
    free_text = build_controlled_free_text(rng, companion_type, budget_amount, diet, pace, avoid)

    preferences = themes[:4]

    request_payload: dict[str, Any] = {
        "city": city,
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=travel_days - 1)).isoformat(),
        "travel_days": travel_days,
        "transportation": transportation,
        "accommodation": accommodation,
        "preferences": preferences,
        "free_text_input": free_text,
        "party": party,
        "budget_constraint": budget_constraint,
    }
    # diet_avoid/diet_positive 必须来自模型实际会看到的 preference_profile 计算路径，
    # 所以在这里对已经生成好的 TripRequest 真跑一遍 infer_food_constraints，而不是
    # 沿用造数侧自己声明的 diet label。
    food_constraints = infer_food_constraints(to_trip_request(request_payload))
    preference_spec = build_preference_control_spec(preferences, free_text, party, pace, avoid, food_constraints)

    return {
        "request_id": format_request_id(index),
        **request_payload,
        "source": "controlled",
        "control_spec": {
            "companion_type": companion_type,
            "city_tier": city_tier,
            "budget_level": budget_level,
            "budget_amount": budget_amount,
            "budget_strictness": budget_constraint["strictness"],
            "pace": pace,
            "diet": diet,
            "avoid": avoid,
            **preference_spec,
        },
    }


def iter_requests(count: int, *, seed: int, date_mode: str) -> list[dict[str, Any]]:
    """批量生成受控请求，index 从 0 到 count-1，seed 固定时逐条可复现。"""
    return [generate_controlled_request(i, seed=seed, date_mode=date_mode) for i in range(count)]


def to_trip_request(item: dict[str, Any]) -> TripRequest:
    """把 generate_controlled_request 的输出 dict 收窄成 TripRequest（丢弃 request_id/source/control_spec）。"""
    payload = {
        key: item[key]
        for key in (
            "city", "start_date", "end_date", "travel_days", "transportation",
            "accommodation", "preferences", "free_text_input", "party", "budget_constraint",
        )
    }
    return TripRequest(**payload)
