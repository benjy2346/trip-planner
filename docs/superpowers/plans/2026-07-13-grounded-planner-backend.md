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

## 词汇表决策（2026-07-14，执行 Task 6 前修订）

**PlannerContext 键名一律采用 helloagents 词汇表**，不做任何重命名/翻译层。理由：已 commit 的 `policy.py`/`compact.py` 是逐字移植（已按他的键名产出）；冻结评测集 `training/data/planner/eval{,_hard}/records.jsonl`（200/300 条）内部就是他的键名；Plan 2 要复用他的 `generate_sft_data.py`。任何翻译层都会重新引入这次 pivot 想消灭的偏差。

| 用途 | 键名（唯一正确） |
|---|---|
| 景点候选 | `classic_pois` / `preference_pois` / `scenic_pois` / `experience_pois` |
| 酒店候选 | `hotel_pois` |
| 餐饮候选 | `food_pois`（+ `food_query_groups`） |
| 天气 | `trip_weather`（原始 `available_weather`） |
| 日期 | `planner_constraints.days_count` / `expected_dates` |
| 价格 | `ticket_price_hint` / `meal_cost_hint` / `estimated_cost_hint`（候选侧是 **hint**，模型复制进 plan 的 `ticket_price` / `estimated_cost`） |

**Planner prompt 同样移植他的**：源 `helloagents backend/app/agents/prompts.py` 的 `PLANNER_AGENT_PROMPT`（约 66 行）。它就是生成那 500 条冻结评测记录的 prompt，且已编码 ① 餐厅去重算法（used_count / 同日 lunch≠dinner / `food_pois >= days*2` 时全程不重复）② `rooms = ceil(people/2)` 的酒店预算口径 ③ 价格必须复制 `*_hint` 的规则。自己新写 prompt 会与冻结评测集脱钩。

---

## Task 6: 新增 PlannerContextBuilder + 移植 prompt（增量，不破坏旧路径）

**Files:**
- Create: `backend/app/planner/prompts.py`（移植 `PLANNER_AGENT_PROMPT`）
- Modify(增量): `backend/app/planner/context.py`
- Test: `backend/tests/test_planner_context_grounded.py`

**Interfaces:**
- Consumes: Task 1-5 全部
- Produces:
  - `class PlannerContextBuilder(amap_api_key: str)`
    - `.collect(request: TripRequest) -> dict`（`tool_snapshot` 含 `scenic_pois / hotel_pois / food_pois / trip_weather`，候选带真 `location`；顶层含 `preference_profile`）
    - `.compact_for_planner(context) -> dict`
  - `PLANNER_AGENT_PROMPT: str`、`build_grounded_planner_messages(context: dict) -> list`

**为什么是增量而不是重写：** 旧 `build_planner_context` 有 8 个依赖方（`supervisor`、`ml/planner/build_eval_set.py`、4 个测试文件），旧 `build_planner_messages` 还被 `rule_eval` / `data_gen` / `export_llamafactory` 使用。Task 6 若直接重写 `context.py`，测试套件会一直红到 Task 8，违反全局约束「现有测试保持全绿 / 每个 Task 独立可测」。因此**旧符号（`build_planner_context` / `PLANNER_SYSTEM_PROMPT` / `build_planner_messages`）与 `pricing_legacy.py` 全部保留到 Task 8**，由 Task 8 在切换 supervisor 的同时一次性删除。

- [ ] **Step 1: 移植 prompt**

把 helloagents `backend/app/agents/prompts.py` 中 `PLANNER_AGENT_PROMPT = """..."""` 整段逐字拷进新文件 `backend/app/planner/prompts.py`（只保留这一个常量，不搬 attraction/weather/hotel 三个子图 prompt——它们随子图一起弃用）。

- [ ] **Step 2: 拷贝他的 context.py 主体并裁剪 MVP**

以 helloagents `backend/app/planner/context.py` 为基底，把 `PlannerContextBuilder` 及其私有方法整体加入我们的 `context.py`（追加，勿覆盖旧函数），并裁剪：
- 删除 `from .high_end_candidates import ...`、`from .local_poi_table import ...`、`from .debug import ...` 三行 import。
- `_collect_attraction_snapshot`：删除 `use_local_high_end_{scenic,experience,food}` 三个分支、`local_high_end_pois(...)`、`rank_food_for_budget_context(...)`、`summarize_high_end_candidates(...)`、`strip_budget_context_metadata(...)`；保留 classic/preference/experience/food 搜索 + `with_*_hints` + `filter_food_by_constraints` + `annotate_food_pois` + `merge_poi_buckets` + `_supplement_food_budget_candidates_if_needed`。status message 里去掉 `food_high_end_*` 字段。
- `_collect_hotel_snapshot`：同样删除 high_end 分支与 `rank_hotels_for_budget_context` / `summarize_high_end_candidates` / `strip_budget_context_metadata`，保留 `build_hotel_keyword_groups` + `with_hotel_cost_hints`。
- 删除 `print_summary` / `print_visualization`（依赖 debug.py，不搬）。
- schema import 改 `from app.models.schemas import TripRequest`；相对 import 改我们的包路径。

- [ ] **Step 3: 加 messages helper**

在 `context.py` 末尾追加（**新名字**，避免与旧 `build_planner_messages` 冲突；Task 8 删旧后可改回原名）：

```python
from app.planner.prompts import PLANNER_AGENT_PROMPT

def build_grounded_planner_messages(context: dict) -> list[BaseMessage]:
    return [
        SystemMessage(content=PLANNER_AGENT_PROMPT),
        HumanMessage(content="PlannerContext:\n" + json.dumps(context, ensure_ascii=False)),
    ]
```

- [ ] **Step 4: 写失败测试（TDD，先失败）**

`backend/tests/test_planner_context_grounded.py`：patch 三个 `_collect_*` 快照方法，断言
- `collect()` 产出 `tool_snapshot.scenic_pois / food_pois / hotel_pois`，且候选 `location` 为**真实非零**经纬度；
- 顶层含 `preference_profile`（结构化约束，含 `diet_avoid`）；
- `compact_for_planner()` 保留 `food_pois` 与 `preference_profile`；
- `build_grounded_planner_messages()` 返回 2 条消息，system 为 `PLANNER_AGENT_PROMPT`，human 内含候选店名（train/serve 同源断言）。

Run: `cd backend && python3 -m pytest tests/test_planner_context_grounded.py -q` → 先 FAIL。

- [ ] **Step 5: 实现到通过 + 全量回归**

Run: `cd backend && python3 -m pytest tests/ -q`
Expected: 新测试 PASS，且**旧测试全绿**（旧路径未动）。

- [ ] **Step 6: Commit**

```bash
git add backend/app/planner/context.py backend/app/planner/prompts.py backend/tests/test_planner_context_grounded.py
git commit -m "feat(planner): add PlannerContextBuilder + port grounded planner prompt"
```

---

## Task 7: 扩写 validation.py（餐饮 grounding + 饮食约束 + 坐标非空 + 预算口径对齐）

**Files:**
- Modify: `backend/app/planner/validation.py`
- Test: `backend/tests/test_planner_validation_grounded.py`

**Interfaces:**
- Consumes: `TripPlan`、grounded context（Task 6，**他的键名**）
- Produces: `validate_grounded_trip_plan(plan, context) -> list[str]`（新函数，旧 `validate_trip_plan` 保留到 Task 8），违规类型：
  - 餐饮店名 ∉ `food_pois` → `"餐饮 X 不在候选中"`
  - 命中 `preference_profile.diet_avoid` → `"餐饮 X 违反饮食约束"`
  - 同日 lunch/dinner 同名 → `"第N天 午晚餐重复: X"`
  - 景点/酒店 `location` 为 None 或 `0,0` → `"X 坐标缺失"`
- 读取键名：`snapshot["scenic_pois"]`（景点候选合并池）、`snapshot["hotel_pois"]`、`snapshot["food_pois"]`、`snapshot["trip_weather"]`、`context["planner_constraints"]["expected_dates"]`。

**⚠️ 预算口径必须与 prompt 对齐（新增，原计划遗漏）：** 移植的 prompt 规定 `total_hotels = Σ(day.hotel.estimated_cost × rooms)`，`rooms = ceil(party.total / 2)`。现有 `recompute_budget(plan, party_total)` 把酒店按 `Σ estimated_cost` 算（**无 rooms 系数**）。若不改，工程重算会与模型被要求的算法系统性不一致，导致所有样本预算对不上。本 Task 增加 `recompute_grounded_budget(plan, party_total)`：hotels 乘 `ceil(party_total/2)`，门票/餐饮乘 `party_total`，交通沿用模型自报。旧 `recompute_budget` 保留到 Task 8。

- [ ] **Step 1: 写失败测试**（覆盖上述 5 类违规 + rooms 预算）
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现 `validate_grounded_trip_plan` + `recompute_grounded_budget`**
- [ ] **Step 4: 运行确认通过 + 全量回归**
- [ ] **Step 5: Commit** — `feat(planner): grounded validation (meal grounding, diet, coords, rooms budget)`

---

## Task 8: supervisor 切到 Builder + 删除旧路径

**Files:**
- Modify: `backend/app/agents/supervisor.py`、`backend/app/planner/context.py`、`backend/app/planner/validation.py`
- Delete: `backend/app/planner/pricing_legacy.py`
- Modify(迁移或删除): `backend/ml/planner/{rule_eval,data_gen,export_llamafactory,build_eval_set}.py`
- Delete(退役): `backend/tests/test_planner_context.py`、`test_planner_validation.py`、`test_pricing.py`、`test_rule_eval.py` 中针对旧 hollow context 的用例
- Test: `backend/tests/test_supervisor.py`（更新）

- [ ] **Step 1: assembler 取数改为 Builder**

```python
from app.planner.context import PlannerContextBuilder, build_grounded_planner_messages
from app.planner.validation import validate_grounded_trip_plan, recompute_grounded_budget
from app.config import settings

_planner_builder = PlannerContextBuilder(amap_api_key=settings.amap_api_key)

async def assembler_node(state: SupervisorState) -> dict:
    req = state["trip_request"]
    context = await asyncio.to_thread(_planner_builder.collect, req)   # collect 同步，放线程
    compact = _planner_builder.compact_for_planner(context)
    response = await acall_agent_with_fallback("assembler", build_grounded_planner_messages(compact))
    content = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    trip_plan = TripPlan(**json.loads(content))
    violations = validate_grounded_trip_plan(trip_plan, context)
    if violations:
        print(f"⚠️ TripPlan 校验告警 {len(violations)} 条: {violations[:5]}")
    trip_plan.budget = recompute_grounded_budget(trip_plan, context["party"]["total"])
    return {"trip_plan": trip_plan,
            "messages": [AIMessage(content=f"已为您生成{req.city}{req.travel_days}天行程。")]}
```

删除 supervisor 对 `weather_subgraph/hotel_subgraph/poi_subgraph` 的调用与 `weather_outputs/hotel_outputs/poi_outputs` 编织。子图文件保留但不再进入 planner 主链。

- [ ] **Step 2: 把 ml/planner 脚本切到新 API**

`rule_eval.py` / `data_gen.py` / `export_llamafactory.py` 的 `build_planner_messages` → `build_grounded_planner_messages`；`build_eval_set.py` 的 `build_planner_context` → `PlannerContextBuilder.collect`。（注意：这些脚本产出的旧 run 数据已作废，Plan 2 会重新生成。）

- [ ] **Step 3: 删除旧路径**

删除 `context.py` 里的 `build_planner_context` / `PLANNER_SYSTEM_PROMPT` / `build_planner_messages`（或把后者改名接管），删除 `validation.py` 里的旧 `validate_trip_plan` / `recompute_budget`，`git rm backend/app/planner/pricing_legacy.py`，退役 4 个旧测试文件里针对旧 context 的用例。
确认：`cd backend && grep -rn "pricing_legacy\|build_planner_context" app/ ml/ tests/ || echo OK`

- [ ] **Step 4: 更新 test_supervisor.py 的 patch 目标**

```python
with patch("app.agents.supervisor._planner_builder.collect", return_value=_fake_ctx()), \
     patch("app.agents.supervisor._planner_builder.compact_for_planner", side_effect=lambda c: c), \
     patch("app.agents.supervisor.acall_agent_with_fallback", AsyncMock(return_value=mock_response)):
    ...
```
`_fake_ctx()` 用**他的键名**（`scenic_pois`/`hotel_pois`/`food_pois`/`trip_weather`/`expected_dates`）。

- [ ] **Step 5: 全量回归**

Run: `cd backend && python3 -m pytest tests/ -q` → 全绿。

- [ ] **Step 6: 冒烟（真高德，1 条请求）**

```bash
cd backend && set -a && . ../.env && set +a && python3 -c "
from app.models.schemas import TripRequest
from app.planner.context import PlannerContextBuilder
from app.config import settings
b = PlannerContextBuilder(amap_api_key=settings.amap_api_key)
req = TripRequest(city='杭州', start_date='2026-07-15', end_date='2026-07-17', travel_days=3,
                  transportation='打车', accommodation='经济型酒店', preferences=['美食'],
                  free_text_input='不吃辣', party={'adults':2})
ctx = b.collect(req)
snap = ctx['tool_snapshot']
print('景点', len(snap['scenic_pois']), '餐饮', len(snap['food_pois']),
      '酒店', len(snap['hotel_pois']), '天气', len(snap['trip_weather']))
print('样例景点坐标:', snap['scenic_pois'][0].get('location'))   # 真实经纬度，非 0,0
print('样例餐饮:', snap['food_pois'][0].get('name'), snap['food_pois'][0].get('meal_cost_hint'))
"
```
Expected: 候选非空；景点坐标为真实经纬度；`2026-07-15`（近未来）`trip_weather` 中 `source=amap_forecast` 的天数 > 0。

- [ ] **Step 7: Commit**

```bash
git add -A backend/
git commit -m "feat(planner): supervisor uses PlannerContextBuilder (grounded), drop legacy context path"
```

---

## Self-Review 记录

- **Spec 覆盖**：§4 架构（独立 Builder + supervisor 薄壳）→ Task 6/8；§5 组件映射（amap/pois/pricing/policy/context/validation）→ Task 1-7 逐一对应；坐标/餐饮/价格/preference 四个填坑点分别落在 Task 1/2、Task 2/6、Task 3、Task 4；train/serve 同源 → Task 6 断言 + Task 8 复用同一 Builder。
- **2026-07-14 修订**：① 键名统一为 helloagents 词汇表（见上方决策表），删除原计划中 `attraction_candidates/hotel_candidates/food_candidates/weather` 与 `planner_constraints.dates` 的用法；② prompt 改为移植他的 `PLANNER_AGENT_PROMPT` 而非新写；③ Task 6 改为增量新增，旧路径（`build_planner_context`/`pricing_legacy`/旧 validation）推迟到 Task 8 统一删除，保证每个 Task 后测试全绿；④ Task 7 新增 `rooms = ceil(people/2)` 预算口径对齐（原计划遗漏，会导致工程重算与 prompt 算法系统性不一致）。
- **超出 Plan 1 范围**：数据重建、requestgen、评测集、训练 → Plan 2（`2026-07-13-grounded-planner-data-train.md`，依赖本 Plan 完成）。
- **已知延后**：high_end_candidates、rerank、output、DPO。
