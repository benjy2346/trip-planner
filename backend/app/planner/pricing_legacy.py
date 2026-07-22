"""合成价签：地图 API 不返回房价，用「酒店档次 × 城市层级」给候选贴一个可复现的估价。

定位：这不是真实房价预测，而是**成本信号**——价签写在候选快照里（输入侧），模型只负责
照抄并据此把总额压进预算（受约束选择），自身从不生成价格。真实价源接入时替换本模块即可，
模型学到的预算行为不变。
"""

# 城市层级：仅覆盖 requestgen 的候选城市；系数是相对量级，非真实房价。
CITY_TIER = {
    "北京": 1, "上海": 1, "广州": 1,
    "杭州": 2, "成都": 2, "南京": 2, "重庆": 2, "苏州": 2, "西安": 2, "厦门": 2,
}
_CITY_MULT = {1: 1.3, 2: 1.0, 3: 0.8}

# 酒店档次基准价（元/晚，单间），按用户 accommodation 偏好。
_HOTEL_BASE = {"经济型酒店": 250, "舒适型酒店": 500, "高档型酒店": 900}
_HOTEL_DEFAULT = 400
# 候选内价差：让"挑便宜的"成为真实预算杠杆，而不是所有候选一个价。
_SPREAD = [0.85, 1.0, 1.15]

# 餐饮单人单餐基准价（元），按餐型。
_MEAL_BASE = {"breakfast": 25, "lunch": 55, "dinner": 85}


def city_tier(city: str) -> int:
    return CITY_TIER.get(city, 2)


def _city_mult(city: str) -> float:
    return _CITY_MULT[city_tier(city)]


def hotel_price(accommodation: str, city: str, index: int = 0) -> int:
    """按 档次 × 城市层级 × 候选内价差 估算每晚房价，取整到 10 元。"""
    base = _HOTEL_BASE.get(accommodation, _HOTEL_DEFAULT)
    price = base * _city_mult(city) * _SPREAD[index % len(_SPREAD)]
    return int(round(price / 10) * 10)


def meal_cost_table(city: str) -> dict:
    """按城市层级给出三餐单人单餐标准价，取整到 5 元。"""
    mult = _city_mult(city)
    return {t: int(round(base * mult / 5) * 5) for t, base in _MEAL_BASE.items()}
