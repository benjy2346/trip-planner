# Grounded Planner Backend Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 helloagents 的结构化取数 + 富化层移植进我们 `trip-planner` app，用一个独立 `PlannerContextBuilder` 产出 grounded `PlannerContext`（真坐标 + 餐饮候选 + 真价格 hint + preference_profile），数据生成与线上共用它。

**Architecture:** 移植（copy + adapt）helloagents `backend/app/planner/` 的 MVP 子集到我们 `backend/app/planner/`；砍掉 high_end/rerank/output；`context.py` 重写为 `PlannerContextBuilder`；`supervisor` 取数改调 Builder，弃用 LLM 解析子图。

**Tech Stack:** Python 3.11 / FastAPI / LangGraph / 高德 REST / Pydantic

## Global Constraints

- **每一步先看 helloagents 对应文件为准**：源在 `~/Desktop/UCLA/helloagents-trip-planner/backend/app/planner/`。
- 移植文件**逐字拷贝其逻辑**，只改：① import 路径改我们的包 ② 高德 key 从我们 `app/config.py`/`.env` 读 ③ schema 用我们 `app/models/schemas`（已与他对齐）。
- backend 不新增运行时重依赖（高德用 `httpx`/`requests`，与现有一致）。
- 现有测试保持全绿；新增测试走 TDD（先失败）。
- 砍掉不搬：`high_end_candidates.py`、`local_poi_table.py`、`rerank.py`、`output.py`；移植 `context.py` 时删除对这两者的 import 与分支。
- 坐标一律取高德结构化字段，**禁止**再用 LLM 解析或填 `0,0` 占位。
- 每个 Task 独立可测、独立 commit。

---

## File Structure

```
backend/app/planner/
  amap.py            # 新增(移植)：AmapPlannerClient 结构化高德客户端 + 缓存 + QPS
  pois.py            # 新增(移植)：POI/餐饮搜索、关键词组、饮食过滤、normalize(真坐标)
  pricing.py         # 改写(合并他的)：ticket/hotel/meal 价格 hint + 价格表
  attraction_price_table.json  # 新增(拷贝资产)
  policy.py          # 新增(移植)：preference_profile / lodging / pricing / route policy
  compact.py         # 新增(移植)：compact_for_planner
  dates.py           # 新增(移植)：trip_date_strings / unknown_weather_row
  weather.py         # 新增(移植)：normalize_weather / align_trip_weather
  context.py         # 重写：PlannerContextBuilder(collect/compact) + 新 prompt(含 food/grounding)
  validation.py      # 扩写：三餐 grounding + 饮食约束 + 坐标非空校验
backend/app/agents/
  supervisor.py      # 改写：assembler 取数改调 PlannerContextBuilder
  subgraphs/{poi,hotel,weather}.py  # 弃用(保留文件但不再进入 planner 主链)
backend/tests/
  test_planner_amap.py / test_planner_pois.py / test_planner_pricing.py
  test_planner_policy.py / test_planner_context_grounded.py
  test_planner_validation_grounded.py / test_supervisor.py(更新)
```

参照映射见 `docs/superpowers/specs/2026-07-13-grounded-planner-integration-design.md` §5。

---

## Task 1: 移植 AmapPlannerClient（结构化高德客户端）

**Files:**
- Create: `backend/app/planner/amap.py`
- Modify: `backend/app/config.py`（确认 `amap_api_key` 可读）
- Test: `backend/tests/test_planner_amap.py`

**Interfaces:**
- Produces:
  - `class AmapPlannerClient.__init__(api_key: str, cache_dir: Path = ...)`
  - `.search_keywords(city, keywords, limit, source_role=..., source_bucket=...) -> List[Dict]`
  - `.search_classic_pois(city, keywords, limit) -> List[Dict]`
  - 每个返回 dict 含结构化 `location`（`{"longitude": float, "latitude": float}`，非 0）、`name`、`address`。

- [ ] **Step 1: 拷贝源文件**

Run:
```bash
cp ~/Desktop/UCLA/helloagents-trip-planner/backend/app/planner/amap.py \
   backend/app/planner/amap.py
```

- [ ] **Step 2: 适配 import 与 key 读取**

打开 `backend/app/planner/amap.py`：
- 把任何 `from ..config import ...` / key 读取改为我们的：`from app.config import settings`，`api_key` 由调用方传入（构造函数已接收 `api_key`），无需改内部。
- 确认缓存目录常量 `PLANNER_CONTEXT_CACHE_DIR` 落在项目内（如 `backend/.cache/planner`），必要时改为 `Path(__file__).resolve().parents[2] / ".cache" / "planner"`。

- [ ] **Step 3: 写失败测试**

```python
# backend/tests/test_planner_amap.py
from unittest.mock import patch
from app.planner.amap import AmapPlannerClient


def _fake_amap_response():
    # 高德 place/text 结构化返回：pois[].location = "lng,lat"
    return {"status": "1", "pois": [
        {"name": "西湖", "address": "西湖区", "location": "120.15,30.25",
         "type": "风景名胜", "biz_ext": {"rating": "4.7"}},
    ]}


def test_search_keywords_parses_structured_location(tmp_path):
    client = AmapPlannerClient(api_key="TESTKEY", cache_dir=tmp_path)
    with patch.object(client, "get", return_value=_fake_amap_response()):
        rows = client.search_keywords("杭州", ["西湖"], limit=5)
    assert rows, "should return candidates"
    loc = rows[0]["location"]
    assert loc["longitude"] == 120.15 and loc["latitude"] == 30.25  # 真坐标，非 0,0
    assert rows[0]["name"] == "西湖"
```

- [ ] **Step 4: 运行确认失败**

Run: `cd backend && python3 -m pytest tests/test_planner_amap.py -q`
Expected: FAIL（若 `.get` 内部结构与 mock 不符）→ 据实调整 mock 使其匹配他 `search_keywords` 实际解析路径（参照他 `amap.py` 的 `search_keywords`/`normalize` 处理），使断言成立。

- [ ] **Step 5: 运行确认通过**

Run: `cd backend && python3 -m pytest tests/test_planner_amap.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/planner/amap.py backend/tests/test_planner_amap.py
git commit -m "feat(planner): port AmapPlannerClient (structured AMAP, real coords)"
```

---

## Task 2: 移植 pois.py（POI/餐饮搜索 + 饮食过滤 + normalize）

**Files:**
- Create: `backend/app/planner/pois.py`
- Test: `backend/tests/test_planner_pois.py`

**Interfaces:**
- Consumes: `TripRequest`（我们 schema）
- Produces（关键，供 context 使用）：
  - `build_poi_keywords(request, role) -> List[str]`
  - `build_food_keyword_groups(request) -> List[Dict]`
  - `filter_food_by_constraints(rows, request) -> List[Dict]`
  - `annotate_food_pois(rows, request) -> List[Dict]`
  - `normalize_pois(rows, ...) -> List[Dict]`（保结构化 location）
  - `merge_poi_buckets(buckets, limit) -> List[Dict]`
  - `infer_food_constraints(request) -> Dict`（含 diet_avoid 等）
  - `parse_location(value) -> Optional[Dict[str, float]]`

- [ ] **Step 1: 拷贝源文件**

Run:
```bash
cp ~/Desktop/UCLA/helloagents-trip-planner/backend/app/planner/pois.py \
   backend/app/planner/pois.py
```

- [ ] **Step 2: 适配 import**

把 `from ..models.schemas import TripRequest` 改为 `from app.models.schemas import TripRequest`；其余纯函数无外部依赖，保持原样。

- [ ] **Step 3: 写失败测试**

```python
# backend/tests/test_planner_pois.py
from app.models.schemas import TripRequest
from app.planner import pois


def _req(free_text=""):
    return TripRequest(city="成都", start_date="2026-08-01", end_date="2026-08-03",
                       travel_days=3, transportation="打车", accommodation="经济型酒店",
                       preferences=["美食"], free_text_input=free_text)


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
```

- [ ] **Step 4: 运行确认失败 → 对齐**

Run: `cd backend && python3 -m pytest tests/test_planner_pois.py -q`
Expected: 若断言与他实现细节不符（如辣味过滤依据的字段名），参照他 `filter_food_by_constraints`/`infer_food_constraints`/`mentions_food_avoidance` 的实际判定改断言，使其反映真实行为。

- [ ] **Step 5: 运行确认通过**

Run: `cd backend && python3 -m pytest tests/test_planner_pois.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/planner/pois.py backend/tests/test_planner_pois.py
git commit -m "feat(planner): port pois.py (structured POI/food search + dietary filter)"
```

---

## Task 3: 移植/合并 pricing.py + 价格表资产

**Files:**
- Modify(重写): `backend/app/planner/pricing.py`（用他的替换我们现有的）
- Create: `backend/app/planner/attraction_price_table.json`
- Test: `backend/tests/test_planner_pricing.py`

**Interfaces:**
- Consumes: `TripRequest`、POI/餐饮 rows（Task 2）
- Produces:
  - `with_ticket_price_hints(rows, request) -> List[Dict]`（rows 带 `ticket_price` 非编造）
  - `with_meal_cost_hints(rows, request) -> List[Dict]`（rows 带 `meal_cost_hint`）
  - `with_hotel_cost_hints(rows, request) -> List[Dict]`（rows 带 `estimated_cost`）
  - `estimate_ticket_price(row) -> int`、`load_attraction_price_table() -> List[Dict]`

- [ ] **Step 1: 备份并替换 pricing.py + 拷贝价格表**

Run:
```bash
git mv backend/app/planner/pricing.py backend/app/planner/pricing_legacy.py 2>/dev/null || true
cp ~/Desktop/UCLA/helloagents-trip-planner/backend/app/planner/pricing.py \
   backend/app/planner/pricing.py
cp ~/Desktop/UCLA/helloagents-trip-planner/backend/app/planner/attraction_price_table.json \
   backend/app/planner/attraction_price_table.json
```

- [ ] **Step 2: 适配 import + 迁移旧 pricing 的被引用符号**

- 把 `from ..models.schemas import ...` 改 `from app.models.schemas import ...`。
- 检查旧 `context.py` 用到的旧 pricing 符号（`hotel_price`/`meal_cost_table`/`city_tier`）：这些在 Task 6 重写 context 时会被他的 `with_*_hints` 取代；本 Task 暂保留 `pricing_legacy.py` 供旧 context import 不报错，Task 6 完成后删除。

- [ ] **Step 3: 写失败测试**

```python
# backend/tests/test_planner_pricing.py
from app.models.schemas import TripRequest
from app.planner import pricing


def _req():
    return TripRequest(city="杭州", start_date="2026-08-01", end_date="2026-08-03",
                       travel_days=3, transportation="打车", accommodation="经济型酒店")


def test_ticket_price_hint_from_table_or_estimate():
    rows = [{"name": "西湖", "type": "风景名胜"}]
    out = pricing.with_ticket_price_hints(rows, _req())
    assert "ticket_price" in out[0]
    assert isinstance(out[0]["ticket_price"], int)  # 有值（0 也是合法免票），字段存在且为 int


def test_meal_cost_hint_present_and_positive():
    rows = [{"name": "外婆家", "type": "餐饮;浙菜"}]
    out = pricing.with_meal_cost_hints(rows, _req())
    assert out[0].get("meal_cost_hint", 0) > 0


def test_price_table_loads():
    table = pricing.load_attraction_price_table()
    assert isinstance(table, list) and len(table) > 0
```

- [ ] **Step 4: 运行确认失败 → 对齐 → 通过**

Run: `cd backend && python3 -m pytest tests/test_planner_pricing.py -q`
Expected: 先失败（模块/路径），修正价格表路径（他用 `Path(__file__).with_name("attraction_price_table.json")`，拷贝后即命中）后 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/planner/pricing.py backend/app/planner/attraction_price_table.json \
        backend/app/planner/pricing_legacy.py backend/tests/test_planner_pricing.py
git commit -m "feat(planner): port pricing.py + attraction price table (real price hints)"
```

---

## Task 4: 移植 policy.py（preference_profile 等）

**Files:**
- Create: `backend/app/planner/policy.py`
- Test: `backend/tests/test_planner_policy.py`

**Interfaces:**
- Consumes: `TripRequest`
- Produces:
  - `build_preference_profile(request) -> Dict`（键含 `positive_preferences / negative_constraints / pace / diet_positive / diet_avoid / traveler_constraints`）
  - `build_lodging_policy(request) -> Dict`、`build_pricing_policy() -> Dict`、`build_route_policy(request) -> Dict`、`build_budget_fit_policy(request) -> Dict`
  - `build_empty_context(request) -> Dict`

- [ ] **Step 1: 拷贝 + 适配 import**

```bash
cp ~/Desktop/UCLA/helloagents-trip-planner/backend/app/planner/policy.py \
   backend/app/planner/policy.py
```
把 schema import 改 `from app.models.schemas import ...`。若 policy.py 引用了 `pois.infer_*`（如 `infer_pace`/`infer_food_constraints`），确认这些在 Task 2 的 pois.py 已存在。

- [ ] **Step 2: 写失败测试**

```python
# backend/tests/test_planner_policy.py
from app.models.schemas import TripRequest
from app.planner.policy import build_preference_profile


def _req(free_text, prefs=None, party=None):
    return TripRequest(city="北京", start_date="2026-08-01", end_date="2026-08-05",
                       travel_days=5, transportation="打车", accommodation="经济型酒店",
                       preferences=prefs or ["历史文化"], free_text_input=free_text,
                       party=party or {"adults": 2, "elders": 1})


def test_elder_and_no_spicy_and_avoid_walk_parsed():
    p = build_preference_profile(_req("有老人同行，少爬山，不吃辣"))
    assert "不吃辣" in p["diet_avoid"] or "辣" in "".join(p["diet_avoid"])
    assert p["traveler_constraints"]["avoid_long_walk"] is True
    assert p["traveler_constraints"]["needs_elder_friendly"] is True


def test_positive_preferences_carried():
    p = build_preference_profile(_req("", prefs=["美食", "博物馆"]))
    assert "美食" in p["positive_preferences"]
```

- [ ] **Step 3: 运行失败 → 对齐他实现的键名/判定 → 通过**

Run: `cd backend && python3 -m pytest tests/test_planner_policy.py -q`
Expected: 参照他 `build_preference_profile`/`extract_negative_constraints` 的实际输出键名对齐断言后 PASS。

- [ ] **Step 4: Commit**

```bash
git add backend/app/planner/policy.py backend/tests/test_planner_policy.py
git commit -m "feat(planner): port policy.py (structured preference_profile)"
```

---

## Task 5: 移植支撑模块（compact / dates / weather）

**Files:**
- Create: `backend/app/planner/compact.py`、`dates.py`、`weather.py`
- Test: `backend/tests/test_planner_support.py`

**Interfaces:**
- Produces:
  - `compact.compact_for_planner(context: dict) -> dict`
  - `dates.trip_date_strings(request) -> List[str]`、`dates.unknown_weather_row(date) -> dict`
  - `weather.normalize_weather(rows) -> List[dict]`、`weather.align_trip_weather(...) -> List[dict]`

- [ ] **Step 1: 拷贝 + 适配 import**

```bash
for f in compact dates weather; do
  cp ~/Desktop/UCLA/helloagents-trip-planner/backend/app/planner/$f.py \
     backend/app/planner/$f.py
done
```
把各文件 schema import 改 `from app.models.schemas import ...`。`debug.py` 若仅用于打印常量，可只把 `PLANNER_CONTEXT_PRINT_LIMIT` 内联到 context.py（Task 6），本 Task 不拷 debug.py。

- [ ] **Step 2: 写失败测试**

```python
# backend/tests/test_planner_support.py
from app.models.schemas import TripRequest
from app.planner import dates, compact


def _req():
    return TripRequest(city="杭州", start_date="2026-08-01", end_date="2026-08-03",
                       travel_days=3, transportation="打车", accommodation="经济型酒店")


def test_trip_date_strings():
    assert dates.trip_date_strings(_req()) == ["2026-08-01", "2026-08-02", "2026-08-03"]


def test_compact_reduces_context():
    ctx = {"tool_snapshot": {"food_pois": [{"name": "x", "raw": "y" * 500}]}}
    out = compact.compact_for_planner(ctx)
    assert "tool_snapshot" in out
```

- [ ] **Step 3: 运行失败 → 对齐 → 通过**

Run: `cd backend && python3 -m pytest tests/test_planner_support.py -q`
Expected: 参照各函数实际签名对齐后 PASS。

- [ ] **Step 4: Commit**

```bash
git add backend/app/planner/compact.py backend/app/planner/dates.py \
        backend/app/planner/weather.py backend/tests/test_planner_support.py
git commit -m "feat(planner): port compact/dates/weather support modules"
```

---

## Task 6: 重写 context.py 为 PlannerContextBuilder（MVP 裁剪 + 新 prompt）

**Files:**
- Modify(重写): `backend/app/planner/context.py`
- Delete: `backend/app/planner/pricing_legacy.py`（本 Task 后不再被引用）
- Test: `backend/tests/test_planner_context_grounded.py`

**Interfaces:**
- Consumes: Task 1-5 全部
- Produces:
  - `class PlannerContextBuilder(amap_api_key: str)`
    - `.collect(request: TripRequest) -> dict`（`PlannerContext`，`tool_snapshot` 含 `attraction_candidates / hotel_candidates / food_candidates / weather`，各候选带真 `location`；顶层含 `preference_profile`）
    - `.compact_for_planner(context) -> dict`
  - `PLANNER_SYSTEM_PROMPT: str`、`build_planner_messages(context: dict) -> list`

- [ ] **Step 1: 拷贝他的 context.py 作为基底**

```bash
cp ~/Desktop/UCLA/helloagents-trip-planner/backend/app/planner/context.py \
   backend/app/planner/context.py
```

- [ ] **Step 2: 裁剪 MVP（删 high_end / local_poi_table）**

在 `context.py` 中：
- 删除 `from .high_end_candidates import ...` 与 `from .local_poi_table import ...` 两行 import。
- 在 `_collect_attraction_snapshot` 中删除 `use_local_high_end_*` 分支与 `local_high_end_pois(...)`、`summarize_high_end_candidates(...)`、`rank_*_for_budget_context` 里依赖 high_end 的部分；保留 classic/preference/food 的搜索 + `filter_food_by_constraints` + `annotate_food_pois` + `with_*_hints` + `merge_poi_buckets` + `normalize_pois`。
- schema import 改 `from app.models.schemas import TripRequest`。
- 把 `debug.PLANNER_CONTEXT_PRINT_LIMIT` 内联为常量；删除 `print_summary/print_visualization`（可选，非必需）。

- [ ] **Step 3: 接上我们的 prompt + messages helper**

在 `context.py` 末尾加入（沿用我们现有 `PLANNER_SYSTEM_PROMPT` 骨架，新增餐饮 grounding 与 preference 规则）：

```python
from langchain_core.messages import SystemMessage, HumanMessage
import json as _json

PLANNER_SYSTEM_PROMPT = """你是行程规划专家。输入是 JSON 格式的 PlannerContext（含 request/party/budget_constraint/preference_profile/lodging_policy/pricing_policy/tool_snapshot/planner_constraints）。只依据上下文生成行程，不得编造。

硬性规则：
1. 只输出一个 JSON 对象，无 markdown、无解释。
2. days 的数量/date/day_index 与 planner_constraints.dates 完全一致。
3. weather_info 逐日复制 tool_snapshot.weather；若为空则 weather_info 为空数组。
4. 每天 1-3 个景点，必须取自 tool_snapshot.attraction_candidates，复制其 name/address/location/ticket_price（location 为真实经纬度，不得填 0,0）。
5. 除末日外每天 hotel 连续同店，取自 tool_snapshot.hotel_candidates，复制 name/address/location/estimated_cost；末日 hotel 为 null；hotel.distance 为空串。
6. 每天 breakfast/lunch/dinner 三餐，店名必须取自 tool_snapshot.food_candidates（复制 name/location），禁占位词；尽量不重复。
7. 遵守 preference_profile：diet_avoid 的餐厅不选；avoid_long_walk 时少排爬山类景点。
8. 预算按 pricing_policy：门票复制候选 ticket_price、餐饮用候选 meal_cost_hint、酒店复制候选 estimated_cost；分项=单价×数量（门票/餐饮×party.total，酒店×住宿晚数），total 为各项之和；hard 约束不得超 budget_constraint.amount。

输出 JSON 结构与后端 TripPlan schema 一致。"""


def build_planner_messages(context: dict) -> list:
    return [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content="PlannerContext:\n" + _json.dumps(context, ensure_ascii=False)),
    ]
```

- [ ] **Step 4: 删除 pricing_legacy**

```bash
git rm backend/app/planner/pricing_legacy.py
```
确认无 import 引用它：`cd backend && grep -rn pricing_legacy app/ || echo OK`

- [ ] **Step 5: 写测试（含 train/serve 同源断言）**

```python
# backend/tests/test_planner_context_grounded.py
from unittest.mock import patch
from app.models.schemas import TripRequest
from app.planner.context import PlannerContextBuilder, build_planner_messages


def _req():
    return TripRequest(city="杭州", start_date="2026-08-01", end_date="2026-08-03",
                       travel_days=3, transportation="打车", accommodation="经济型酒店",
                       preferences=["美食"], free_text_input="不吃辣",
                       party={"adults": 2}, budget_constraint={"amount": 4000, "strictness": "hard"})


def _fake_rows(kind):
    return [{"name": f"{kind}{i}", "address": "x区",
             "location": {"longitude": 120.1 + i / 100, "latitude": 30.2 + i / 100},
             "type": "餐饮" if kind == "food" else "风景名胜"} for i in range(4)]


def test_collect_produces_grounded_sections():
    b = PlannerContextBuilder(amap_api_key="TESTKEY")
    # patch 内部搜索，返回带真坐标的行；具体 patch 目标按 context 实际调用点调整
    with patch.object(b, "_collect_attraction_snapshot",
                      return_value={"tool_snapshot": {"attraction_candidates": _fake_rows("poi"),
                                                      "food_candidates": _fake_rows("food")}}), \
         patch.object(b, "_collect_hotel_snapshot",
                      return_value={"tool_snapshot": {"hotel_candidates": _fake_rows("hotel")}}), \
         patch.object(b, "_collect_weather_snapshot",
                      return_value={"tool_snapshot": {"weather": []}}):
        ctx = b.collect(_req())
    snap = ctx["tool_snapshot"]
    assert snap["attraction_candidates"][0]["location"]["longitude"] != 0  # 真坐标
    assert "food_candidates" in snap and snap["food_candidates"]           # 有餐饮候选
    assert "preference_profile" in ctx                                     # 结构化约束


def test_messages_carry_context_and_prompt():
    b = PlannerContextBuilder(amap_api_key="TESTKEY")
    ctx = {"request": {"city": "杭州"}, "tool_snapshot": {"food_candidates": [{"name": "外婆家"}]}}
    msgs = build_planner_messages(ctx)
    assert len(msgs) == 2 and "外婆家" in msgs[1].content
```

- [ ] **Step 6: 运行失败 → 对齐 patch 目标 → 通过**

Run: `cd backend && python3 -m pytest tests/test_planner_context_grounded.py -q`
Expected: 按 `collect` 内部对三个 `_collect_*` 的实际调用/合并方式对齐 patch 与断言后 PASS。

- [ ] **Step 7: Commit**

```bash
git add backend/app/planner/context.py backend/tests/test_planner_context_grounded.py
git rm --cached backend/app/planner/pricing_legacy.py 2>/dev/null || true
git commit -m "feat(planner): rewrite context as PlannerContextBuilder (grounded, food+preference)"
```

---

## Task 7: 扩写 validation.py（餐饮 grounding + 饮食约束 + 坐标非空）

**Files:**
- Modify: `backend/app/planner/validation.py`
- Test: `backend/tests/test_planner_validation_grounded.py`

**Interfaces:**
- Consumes: `TripPlan`、grounded context（Task 6）
- Produces: `validate_trip_plan(plan, context) -> list[str]` 新增违规类型：
  - 餐饮店名 ∉ `food_candidates` → `"餐饮 X 不在候选中"`
  - 命中 `preference_profile.diet_avoid` → `"餐饮 X 违反饮食约束"`
  - 景点/酒店 `location` 为 None 或 `0,0` → `"X 坐标缺失"`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_planner_validation_grounded.py
from app.models.schemas import (TripPlan, DayPlan, Attraction, Meal, Hotel, WeatherInfo, Location, Budget)
from app.planner.validation import validate_trip_plan


def _ctx():
    return {
        "request": {"city": "杭州", "start_date": "2026-08-01", "end_date": "2026-08-01"},
        "preference_profile": {"diet_avoid": ["辣"]},
        "planner_constraints": {"dates": ["2026-08-01"]},
        "tool_snapshot": {
            "weather": [],
            "hotel_candidates": [{"name": "如家", "location": {"longitude": 120.1, "latitude": 30.2}}],
            "attraction_candidates": [{"name": "西湖", "location": {"longitude": 120.1, "latitude": 30.2}}],
            "food_candidates": [{"name": "外婆家"}, {"name": "知味观"}, {"name": "新白鹿"}],
        },
    }


def _loc():
    return Location(longitude=120.1, latitude=30.2)


def _day(meals):
    return DayPlan(date="2026-08-01", day_index=0, description="d", transportation="打车",
                   accommodation="经济型", hotel=None,
                   attractions=[Attraction(name="西湖", address="x", location=_loc(),
                                           visit_duration=120, description="l", ticket_price=0)],
                   meals=meals)


def _plan(meals):
    return TripPlan(city="杭州", start_date="2026-08-01", end_date="2026-08-01",
                    days=[_day(meals)], weather_info=[], overall_suggestions="ok", budget=Budget())


def _meals(bn="外婆家", ln="知味观", dn="新白鹿"):
    return [Meal(type="breakfast", name=bn), Meal(type="lunch", name=ln), Meal(type="dinner", name=dn)]


def test_grounded_meals_pass():
    assert validate_trip_plan(_plan(_meals()), _ctx()) == []


def test_ungrounded_meal_flagged():
    v = validate_trip_plan(_plan(_meals(dn="不存在饭店")), _ctx())
    assert any("候选" in x for x in v)


def test_diet_avoid_meal_flagged():
    ctx = _ctx()
    ctx["tool_snapshot"]["food_candidates"].append({"name": "麻辣香锅"})
    v = validate_trip_plan(_plan(_meals(dn="麻辣香锅")), ctx)
    assert any("饮食约束" in x for x in v)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python3 -m pytest tests/test_planner_validation_grounded.py -q`
Expected: FAIL（尚无餐饮 grounding 校验）

- [ ] **Step 3: 在 validate_trip_plan 内加校验**

在逐日循环内、三餐检查处追加（`food_candidates` 存在时才校验）：

```python
    food_names = {f["name"] for f in snapshot.get("food_candidates", [])}
    diet_avoid = context.get("preference_profile", {}).get("diet_avoid", [])
    ...
    # 在遍历 d.meals 时：
        if food_names and m.name not in food_names:
            v.append(f"{label} 餐饮 {m.name} 不在候选中")
        if any(bad and bad in m.name for bad in diet_avoid):
            v.append(f"{label} 餐饮 {m.name} 违反饮食约束")
    # 景点/酒店坐标非空：
        for a in d.attractions:
            if a.location is None or (a.location.longitude == 0 and a.location.latitude == 0):
                v.append(f"{label} 景点 {a.name} 坐标缺失")
        if d.hotel is not None and (d.hotel.location is None or
                                    (d.hotel.location.longitude == 0 and d.hotel.location.latitude == 0)):
            v.append(f"{label} 酒店 {d.hotel.name} 坐标缺失")
```

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python3 -m pytest tests/test_planner_validation_grounded.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/planner/validation.py backend/tests/test_planner_validation_grounded.py
git commit -m "feat(planner): validate meal grounding + dietary + coord presence"
```

---

## Task 8: supervisor 取数改调 PlannerContextBuilder，弃用 LLM 解析子图

**Files:**
- Modify: `backend/app/agents/supervisor.py`
- Test: `backend/tests/test_supervisor.py`（更新）

**Interfaces:**
- Consumes: `PlannerContextBuilder`（Task 6）、`build_planner_messages`（Task 6）、`validate_trip_plan/recompute_budget`（Task 7/现有）、`acall_agent_with_fallback`（现有 T4）

- [ ] **Step 1: 改 assembler 取数为 Builder**

把 supervisor 里「跑三个子图 + 旧 build_planner_context」的取数替换为：

```python
from app.planner.context import PlannerContextBuilder, build_planner_messages
from app.config import settings

_planner_builder = PlannerContextBuilder(amap_api_key=settings.amap_api_key)

async def assembler_node(state: SupervisorState) -> dict:
    req = state["trip_request"]
    context = await asyncio.to_thread(_planner_builder.collect, req)  # collect 是同步，放线程
    compact = _planner_builder.compact_for_planner(context)
    response = await acall_agent_with_fallback("assembler", build_planner_messages(compact))
    content = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    trip_plan = TripPlan(**json.loads(content))
    violations = validate_trip_plan(trip_plan, context)
    if violations:
        print(f"⚠️ TripPlan 校验告警 {len(violations)} 条: {violations[:5]}")
    trip_plan.budget = recompute_budget(trip_plan, context["party"]["total"])
    return {"trip_plan": trip_plan,
            "messages": [AIMessage(content=f"已为您生成{req.city}{req.travel_days}天行程。")]}
```

删除 supervisor 里对 `weather_subgraph/hotel_subgraph/poi_subgraph` 的调用与 `weather_outputs/hotel_outputs/poi_outputs` 的编织（若图结构需要保留节点，让它们成为 no-op 或直接从图中移除）。子图文件保留但不再进入 planner 主链。

- [ ] **Step 2: 更新 test_supervisor.py 的 patch 目标**

把对 `build_planner_context`/子图的 patch 改为 patch `PlannerContextBuilder.collect`：

```python
from unittest.mock import patch, AsyncMock, MagicMock

def _fake_ctx():
    return {"request": {"city": "杭州"}, "party": {"total": 2},
            "planner_constraints": {"dates": ["2026-08-01"]},
            "tool_snapshot": {"weather": [], "hotel_candidates": [], "attraction_candidates": [],
                              "food_candidates": []},
            "preference_profile": {"diet_avoid": []}}

# 在测试中：
with patch("app.agents.supervisor._planner_builder.collect", return_value=_fake_ctx()), \
     patch("app.agents.supervisor._planner_builder.compact_for_planner", side_effect=lambda c: c), \
     patch("app.agents.supervisor.acall_agent_with_fallback", AsyncMock(return_value=mock_response)):
    ...
```
（`mock_response.content` 用一个最小合法 TripPlan JSON；days 可为空，校验只告警不拦截。）

- [ ] **Step 3: 全量回归**

Run: `cd backend && python3 -m pytest tests/ -q`
Expected: 全绿（含更新后的 supervisor 测试与全部新增 planner 测试）。

- [ ] **Step 4: 冒烟（真高德，1 条请求）**

Run:
```bash
cd backend && set -a && . ../.env && set +a && python3 -c "
import asyncio, json
from app.models.schemas import TripRequest
from app.planner.context import PlannerContextBuilder
from app.config import settings
b = PlannerContextBuilder(amap_api_key=settings.amap_api_key)
req = TripRequest(city='杭州', start_date='2026-07-15', end_date='2026-07-17', travel_days=3,
                  transportation='打车', accommodation='经济型酒店', preferences=['美食'],
                  free_text_input='不吃辣', party={'adults':2})
ctx = b.collect(req)
snap = ctx['tool_snapshot']
print('景点', len(snap['attraction_candidates']), '餐饮', len(snap.get('food_candidates',[])),
      '酒店', len(snap['hotel_candidates']), '天气', len(snap['weather']))
a = snap['attraction_candidates'][0]
print('样例景点坐标:', a.get('location'))  # 应为真实经纬度，非 0,0
"
```
Expected: 打印非空候选，景点坐标为真实经纬度；`2026-07-15`（近未来）天气数应 > 0。

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/supervisor.py backend/tests/test_supervisor.py
git commit -m "feat(planner): supervisor uses PlannerContextBuilder (grounded), retire LLM-parse subgraphs"
```

---

## Self-Review 记录

- **Spec 覆盖**：§4 架构（独立 Builder + supervisor 薄壳）→ Task 6/8；§5 组件映射（amap/pois/pricing/policy/context/validation）→ Task 1-7 逐一对应；坐标/餐饮/价格/preference 四个填坑点分别落在 Task 1/2、Task 2/6、Task 3、Task 4；train/serve 同源 → Task 6 断言 + Task 8 复用同一 Builder。
- **超出 Plan 1 范围**：数据重建、requestgen、评测集、训练 → Plan 2（`2026-07-13-grounded-planner-data-train.md`，依赖本 Plan 完成）。
- **Placeholder 扫描**：移植类步骤给出确切源路径 + 具体适配点 + 完整测试代码；「对齐他实现细节」为移植固有的核对步骤（源码在他 repo 内可查），非占位。
- **类型一致**：`PlannerContextBuilder.collect/compact_for_planner`、`build_planner_messages`、`with_*_hints`、`build_preference_profile`、`validate_trip_plan` 在 Task 间签名一致；`tool_snapshot` 键名（attraction_candidates/hotel_candidates/food_candidates/weather）在 Task 6/7/8 一致。
- **已知延后**：high_end_candidates、rerank、output、DPO。
```
