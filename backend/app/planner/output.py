"""Planner 输出渲染模块的最小移植子集。

完整版 output.py（他仓库里 570 行的渲染模块）不在本次 MVP 移植范围内。这里搬运两组
被下游依赖的纯函数，行为对齐他的原版：
- `pricing.py` 依赖的 `normalize_poi_name` + 归一化常量；
- `validation.py`（Task 7）依赖的 grounding 谓词：`name_in_candidates`（模糊别名匹配）、
  `is_invalid_hotel_name`、`is_placeholder_hotel_distance`、`is_placeholder_meal_name`、
  `is_lodging_breakfast_meal`。
关键词表按数据原样搬运——改动它们等于改变判定行为。
"""

import re
import unicodedata
from typing import List

TRADITIONAL_POI_CHAR_MAP = str.maketrans(
    {
        "發": "发",
        "髮": "发",
        "隨": "随",
        "園": "园",
        "館": "馆",
        "舘": "馆",
        "樓": "楼",
        "廳": "厅",
        "廚": "厨",
        "齋": "斋",
        "寶": "宝",
        "貝": "贝",
        "龍": "龙",
        "鳳": "凤",
        "麗": "丽",
        "樂": "乐",
        "麥": "麦",
        "麵": "面",
        "魚": "鱼",
        "鮮": "鲜",
        "雞": "鸡",
        "鴨": "鸭",
        "鵝": "鹅",
        "滷": "卤",
        "鹵": "卤",
        "燒": "烧",
        "鍋": "锅",
        "湯": "汤",
        "粵": "粤",
        "廣": "广",
        "東": "东",
        "雲": "云",
        "臺": "台",
        "灣": "湾",
        "門": "门",
        "閣": "阁",
        "舊": "旧",
        "藝": "艺",
        "鄉": "乡",
        "鎮": "镇",
        "縣": "县",
        "區": "区",
        "國": "国",
        "華": "华",
        "順": "顺",
        "親": "亲",
        "實": "实",
        "師": "师",
        "壹": "一",
        "貳": "二",
        "參": "三",
        "叁": "三",
        "萬": "万",
        "點": "点",
        "號": "号",
        "會": "会",
        "軒": "轩",
        "莊": "庄",
        "餅": "饼",
        "餃": "饺",
        "飯": "饭",
        "飲": "饮",
        "餛": "馄",
        "飩": "饨",
        "鱔": "鳝",
        "蝦": "虾",
        "醬": "酱",
        "醃": "腌",
        "臘": "腊",
        "衚": "胡",
        "鬍": "胡",
        "裏": "里",
        "裡": "里",
        "內": "内",
        "長": "长",
        "慶": "庆",
        "賓": "宾",
        "貴": "贵",
        "陽": "阳",
        "寧": "宁",
        "蘇": "苏",
        "廈": "厦",
        "錦": "锦",
        "橋": "桥",
        "頭": "头",
        "島": "岛",
        "濱": "滨",
        "獅": "狮",
        "龜": "龟",
        "鷺": "鹭",
        "遙": "遥",
        "遊": "游",
        "覽": "览",
        "戲": "戏",
        "劇": "剧",
        "場": "场",
        "舖": "铺",
        "鋪": "铺",
        "車": "车",
        "馬": "马",
        "騰": "腾",
        "騎": "骑",
        "範": "范",
        "豐": "丰",
        "億": "亿",
        "銀": "银",
        "鐵": "铁",
        "鉑": "铂",
        "鑽": "钻",
    }
)

POI_BRACKET_CONTENT_RE = re.compile(r"[\(\[\{<【（［｛].*?[\)\]\}>】）］｝]")
POI_KEEP_ALNUM_CJK_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")


POI_NAME_SUFFIXES = [
    "历史文化街区", "风景名胜区", "旅游景区", "森林公园", "湿地公园", "研究基地",
    "步行街", "文化街区", "商业街区", "博物馆", "博物院", "纪念馆", "美术馆",
    "艺术馆", "科技馆", "展览馆", "风景区", "旅游区", "古街区", "景区", "公园",
    "古街", "老街", "古镇", "基地", "中心", "广场",
]

POI_CITY_PREFIXES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "西安", "南京", "苏州",
    "厦门", "青岛", "武汉", "长沙", "昆明", "大理", "丽江", "桂林", "三亚", "哈尔滨",
]

INVALID_HOTEL_NAME_MARKERS = [
    "无住宿", "无需住宿", "不用住宿", "不住宿", "无酒店", "不用酒店", "返程", "回程", "离店",
]
INVALID_HOTEL_NAMES = {"无", "无住宿", "无需住宿", "不住宿", "无酒店", "返程", "当天返程", "回程"}

PLACEHOLDER_MEAL_NAMES = {
    "早餐", "午餐", "晚餐", "早餐推荐", "午餐推荐", "晚餐推荐", "餐饮推荐",
    "本地早餐", "本地午餐", "本地晚餐", "当地早餐", "当地午餐", "当地晚餐",
    "特色早餐", "特色午餐", "特色晚餐", "特色餐厅",
}

HOTEL_BREAKFAST_MARKERS = ["酒店早餐", "酒店自助早餐", "民宿早餐", "客栈早餐", "住宿早餐"]

HOTEL_DISTANCE_PLACEHOLDER_NAMES = {
    "距离景点2公里", "距景点2公里", "距离主要景点2公里", "距主要景点2公里",
    "距离当日景点2公里", "距当日景点2公里",
}


def normalize_poi_name(name: str) -> str:
    """用于日志告警的POI名称归一化，不改变模型最终输出。"""
    text = unicodedata.normalize("NFKC", str(name or "")).strip().lower()
    text = POI_BRACKET_CONTENT_RE.sub("", text)
    text = text.translate(TRADITIONAL_POI_CHAR_MAP)
    text = text.replace("&", "and")
    text = POI_KEEP_ALNUM_CJK_RE.sub("", text)
    return text


def poi_name_aliases(name: str) -> List[str]:
    """生成少量可解释别名，减少工具候选命中判定里的假阴性。

    去掉城市前缀（如「杭州西湖」→「西湖」）和常见景点后缀（如「西湖风景名胜区」→「西湖」），
    这样模型写全称、候选是简称时仍能判为 grounded。
    """
    base = normalize_poi_name(name)
    if not base:
        return []

    aliases = {base}
    for prefix in POI_CITY_PREFIXES:
        if base.startswith(prefix) and len(base) > len(prefix) + 1:
            aliases.add(base[len(prefix):])
    for alias in list(aliases):
        for suffix in POI_NAME_SUFFIXES:
            if alias.endswith(suffix) and len(alias) > len(suffix) + 1:
                aliases.add(alias[: -len(suffix)])
    return sorted(alias for alias in aliases if len(alias) >= 2)


def name_in_candidates(name: str, candidates: List[str]) -> bool:
    """宽松名称匹配，兼容「锦里古街」vs「锦里」这类别名/子串。"""
    name_aliases = poi_name_aliases(name)
    if not name_aliases:
        return False
    for candidate in candidates:
        for left in name_aliases:
            for right in poi_name_aliases(candidate):
                if left == right or left in right or right in left:
                    return True
    return False


def is_invalid_hotel_name(name: str) -> bool:
    """识别模型把「无住宿/返程」写进 hotel.name 的脏数据。"""
    normalized = normalize_poi_name(name)
    if not normalized:
        return True
    if normalized in {normalize_poi_name(item) for item in INVALID_HOTEL_NAMES}:
        return True
    return any(marker in normalized for marker in INVALID_HOTEL_NAME_MARKERS)


def is_placeholder_hotel_distance(distance: str) -> bool:
    """识别模型编造的酒店距离占位（无真实路线工具时应为空串）。"""
    text = str(distance or "").strip()
    if not text:
        return False
    normalized = normalize_poi_name(text)
    if normalized in {normalize_poi_name(item) for item in HOTEL_DISTANCE_PLACEHOLDER_NAMES}:
        return True
    return bool(re.fullmatch(r"(距离|距)(当日|主要|周边|附近)?景点约?\d+(公里|km|米|m)", normalized))


def is_placeholder_meal_name(name: str) -> bool:
    """识别模型输出的餐饮占位词（满足 schema 但无真实餐饮内容）。"""
    normalized = normalize_poi_name(name)
    if not normalized:
        return True
    if normalized in {normalize_poi_name(item) for item in PLACEHOLDER_MEAL_NAMES}:
        return True
    return bool(re.fullmatch(r"第?\d+天?(早餐|午餐|晚餐)", normalized))


def is_hotel_breakfast_name(name: str) -> bool:
    """酒店/民宿早餐不是 food_pois，但可以作为早餐来源。"""
    normalized = normalize_poi_name(name)
    return any(normalize_poi_name(marker) in normalized for marker in HOTEL_BREAKFAST_MARKERS)


def is_lodging_breakfast_meal(name: str, meal_type: str) -> bool:
    """只有早餐餐次可以把住宿早餐视为已 grounding。"""
    return str(meal_type or "").strip().lower() == "breakfast" and is_hotel_breakfast_name(name)


def meal_diversity_key(name: str) -> str:
    """餐饮多样性去重 key：去掉分店括号，让同品牌分店按同一家统计。"""
    text = str(name or "").strip()
    if not text:
        return ""
    for marker in ("(", "（"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return "".join(text.split()).lower()
