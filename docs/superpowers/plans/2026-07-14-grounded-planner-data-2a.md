# Grounded Planner 训练数据生成（Plan 2a）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 参照 helloagents `generate_sft_data.py` 写我们自己的 grounded 训练数据生成器，产出 `(PlannerContext → TripPlan)` 的 sharegpt 训练样本，供 Plan 2b 微调。

**Architecture:** 全量重写请求生成器（含 control_spec）+ 移植历史天气 Builder + teacher 循环 + grounded 校验清洗 + sharegpt 导出。取数用 Plan 1 的 `PlannerContextBuilder`（子类加历史天气），消息用 `build_grounded_planner_messages`，清洗判据用 `validate_grounded_trip_plan`。旧 `requestgen.py` / `data_gen.py` 让位（不删，legacy 仍引用）。

**Tech Stack:** Python 3.11 / httpx / LangChain (`ChatOpenAI` for DeepSeek teacher) / asyncio / pytest

## Global Constraints

- **参照 helloagents，写我们自己的，不逐字照抄**；源在 `~/Desktop/UCLA/helloagents-trip-planner/`。prompt 文本是唯一逐字例外（Plan 1 已移植）。
- schema import 一律 `from app.models.schemas import ...`；取数用 `app.planner.context.PlannerContextBuilder`。
- 键名用 helloagents 词汇表（`scenic_pois`/`food_pois`/`hotel_pois`/`trip_weather`/`expected_dates`/`*_hint`）。
- 新代码放 `backend/ml/planner/datagen/`；不改动 Plan 1 的 `app/planner/*`。
- 现有 134 个测试保持全绿；新增测试走 TDD（先失败）。
- teacher = DeepSeek，长超时 300s；请求 seed 必须与评测集 seed 区分。
- **历史天气仅对过去行程生效**（`is_past_trip`：end_date < today）；训练请求用 `--date-mode mixed`（过去 + 近未来），不要全远期，否则天气段为空。
- 长跑一律 `nohup + caffeinate + disown`（本地 heavy 任务会被 harness 回收）。

---

## File Structure

```
backend/ml/planner/datagen/
  __init__.py
  historical_weather.py   # 移植：is_past_trip / fetch_historical_trip_weather / 城市坐标表 / 天气码映射 / 缓存
  context_builder.py      # DataGenPlannerContextBuilder(PlannerContextBuilder) + throttle_open_meteo_call
  requests.py             # 全量重写：受控请求生成器 + control_spec（controlled source）
  leakage.py              # eval_signature（作用于 TripRequest）+ load_eval_signatures
  generate.py             # 主脚本：teacher 循环 + manifest/resume/usage/容错 + 记录组装
  export.py               # make_lf_row + write_llamafactory_files（sharegpt train/val）
backend/tests/
  test_datagen_historical_weather.py
  test_datagen_context_builder.py
  test_datagen_requests.py
  test_datagen_leakage.py
  test_datagen_generate.py
  test_datagen_export.py
backend/ml/planner/eval/records.jsonl        # 覆盖为他的 200 条 grounded 评测记录
backend/ml/planner/eval_hard/records.jsonl   # 覆盖为他的 300 条 grounded 评测记录
```

参照源映射：
- `historical_weather.py` ← 他 `training/scripts/eval/historical_weather.py`（303 行）
- `context_builder.py` ← 他 `generate_sft_data.py:376-426`（throttle + DataGenPlannerContextBuilder）
- `requests.py` ← 他 `generate_sft_data.py:459-1165`（controlled source 相关；不搬 template/llm source）
- `generate.py` ← 他 `generate_sft_data.py` 主循环 + 我们 `ml/planner/data_gen.py` 脚手架
- `export.py` ← 他 `generate_sft_data.py:1594-1650` + 我们 `ml/planner/export_llamafactory.py`

---

## Task 1: 移植 historical_weather.py（Open-Meteo 历史天气）

**Files:**
- Create: `backend/ml/planner/datagen/__init__.py`（空）
- Create: `backend/ml/planner/datagen/historical_weather.py`
- Test: `backend/tests/test_datagen_historical_weather.py`

**Interfaces:**
- Produces:
  - `is_past_trip(request: TripRequest) -> bool`
  - `fetch_historical_trip_weather(request: TripRequest) -> list[dict]`（每行含 `date/day_weather/night_weather/day_temp/night_temp/source="open_meteo_archive"`）
  - `TOURISM_CITY_COORDS: dict[str, tuple[float,float]]`、`WEATHER_CODE_TEXT: dict[int,str]`

- [ ] **Step 1: 拷贝并适配源文件**

从他 `training/scripts/eval/historical_weather.py` 逐段搬到 `backend/ml/planner/datagen/historical_weather.py`，只改三处：
- `PROJECT_ROOT = Path(__file__).resolve().parents[3]` 改为指向我们 `backend/` 的相对层级（`parents[2]` = backend/），缓存目录改 `backend/ml/planner/data/cache/open_meteo_archive`。
- import 保持 `from app.planner.dates import trip_date_strings` 和 `from app.models.schemas import TripRequest`（我们已有）。
- 用 `httpx`（我们已依赖）。
保留：`TOURISM_CITY_COORDS`、`WEATHER_CODE_TEXT`、`fetch_open_meteo_archive`、`normalize_open_meteo_daily`、`is_past_trip`、`fetch_historical_trip_weather`、缓存读写。

- [ ] **Step 2: 写失败测试**

```python
# backend/tests/test_datagen_historical_weather.py
from unittest.mock import patch
from datetime import date, timedelta
from app.models.schemas import TripRequest
from ml.planner.datagen.historical_weather import is_past_trip, fetch_historical_trip_weather


def _req(city="杭州", start="2020-04-01", days=3):
    end = (date.fromisoformat(start) + timedelta(days=days - 1)).isoformat()
    return TripRequest(city=city, start_date=start, end_date=end, travel_days=days,
                       transportation="打车", accommodation="经济型酒店", preferences=[],
                       party={"adults": 2}, budget_constraint={"amount": 3000, "strictness": "soft"})


def test_is_past_trip_true_for_old_dates():
    assert is_past_trip(_req(start="2020-04-01")) is True


def test_is_past_trip_false_for_future_dates():
    future = (date.today() + timedelta(days=30)).isoformat()
    assert is_past_trip(_req(start=future)) is False


def test_fetch_returns_empty_for_unknown_city():
    assert fetch_historical_trip_weather(_req(city="不存在城")) == []


def test_fetch_normalizes_open_meteo_daily():
    fake = {"daily": {"time": ["2020-04-01", "2020-04-02", "2020-04-03"],
                      "weather_code": [0, 61, 3],
                      "temperature_2m_max": [20.1, 18.0, 16.5],
                      "temperature_2m_min": [10.0, 9.2, 8.1]}}
    with patch("ml.planner.datagen.historical_weather.fetch_open_meteo_archive", return_value=fake):
        rows = fetch_historical_trip_weather(_req())
    assert len(rows) == 3
    assert rows[0]["day_weather"] == "晴"        # weather_code 0 → 晴
    assert rows[0]["source"] == "open_meteo_archive"
    assert int(rows[0]["day_temp"]) == 20
```

- [ ] **Step 3: 运行确认失败**

Run: `cd backend && python3 -m pytest tests/test_datagen_historical_weather.py -q -p no:warnings`
Expected: FAIL（模块或函数不存在）。若断言字段名与他实现不符，打开他 `historical_weather.py` 的 `normalize_open_meteo_daily` 对齐字段名。

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python3 -m pytest tests/test_datagen_historical_weather.py -q -p no:warnings`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/planner/datagen/__init__.py backend/ml/planner/datagen/historical_weather.py backend/tests/test_datagen_historical_weather.py
git commit -m "feat(datagen): port Open-Meteo historical weather module"
```

---

## Task 2: DataGenPlannerContextBuilder（历史天气 override + 限流）

**Files:**
- Create: `backend/ml/planner/datagen/context_builder.py`
- Test: `backend/tests/test_datagen_context_builder.py`

**Interfaces:**
- Consumes: `PlannerContextBuilder`（Plan 1）、Task 1 的 `is_past_trip`/`fetch_historical_trip_weather`
- Produces:
  - `throttle_open_meteo_call() -> None`
  - `class DataGenPlannerContextBuilder(PlannerContextBuilder)`，`__init__(amap_api_key, historical_weather_provider)`，override `_collect_weather_snapshot`

- [ ] **Step 1: 拷贝并适配**

参照他 `generate_sft_data.py:376-426`，写 `context_builder.py`：`throttle_open_meteo_call`（模块级锁 + `OPEN_METEO_MIN_INTERVAL_SECONDS`）+ `DataGenPlannerContextBuilder`。override 逻辑：`historical_weather_provider == "open-meteo" and is_past_trip(request)` 时，throttle → `fetch_historical_trip_weather` → 有行则返回 `{"tool_snapshot": {"available_weather": rows, "trip_weather": rows}, "status": ...}`（3 次重试，失败回退 `super()._collect_weather_snapshot`）；否则直接 `super()`。

- [ ] **Step 2: 写失败测试**

```python
# backend/tests/test_datagen_context_builder.py
from unittest.mock import patch
from datetime import date, timedelta
from app.models.schemas import TripRequest
from ml.planner.datagen.context_builder import DataGenPlannerContextBuilder


def _req(start, days=3):
    end = (date.fromisoformat(start) + timedelta(days=days - 1)).isoformat()
    return TripRequest(city="杭州", start_date=start, end_date=end, travel_days=days,
                       transportation="打车", accommodation="经济型酒店", preferences=[],
                       party={"adults": 2}, budget_constraint={"amount": 3000, "strictness": "soft"})


def test_past_trip_uses_open_meteo():
    b = DataGenPlannerContextBuilder(amap_api_key="X", historical_weather_provider="open-meteo")
    rows = [{"date": "2020-04-01", "day_weather": "晴", "source": "open_meteo_archive"}]
    with patch("ml.planner.datagen.context_builder.fetch_historical_trip_weather", return_value=rows), \
         patch("ml.planner.datagen.context_builder.throttle_open_meteo_call"):
        snap = b._collect_weather_snapshot(_req("2020-04-01"))
    assert snap["tool_snapshot"]["trip_weather"] == rows


def test_future_trip_falls_back_to_super():
    b = DataGenPlannerContextBuilder(amap_api_key="X", historical_weather_provider="open-meteo")
    future = (date.today() + timedelta(days=30)).isoformat()
    with patch("app.planner.context.PlannerContextBuilder._collect_weather_snapshot",
               return_value={"tool_snapshot": {"trip_weather": []}, "status": {"ok": True, "message": "amap"}}) as sup:
        b._collect_weather_snapshot(_req(future))
    sup.assert_called_once()
```

- [ ] **Step 3: 运行失败 → 实现 → 通过**

Run: `cd backend && python3 -m pytest tests/test_datagen_context_builder.py -q -p no:warnings`
Expected: 先 FAIL，实现后 PASS。

- [ ] **Step 4: Commit**

```bash
git add backend/ml/planner/datagen/context_builder.py backend/tests/test_datagen_context_builder.py
git commit -m "feat(datagen): DataGenPlannerContextBuilder with historical-weather override"
```

---

## Task 3: 受控请求生成器 + control_spec（全量重写 controlled source）

**Files:**
- Create: `backend/ml/planner/datagen/requests.py`
- Test: `backend/tests/test_datagen_requests.py`

**Interfaces:**
- Produces:
  - `generate_controlled_request(index: int, *, seed: int, date_mode: str) -> dict`（返回含 `city/start_date/end_date/travel_days/transportation/accommodation/preferences/free_text_input/party/budget_constraint/control_spec` 的 dict）
  - `iter_requests(count: int, *, seed: int, date_mode: str) -> list[dict]`
  - `to_trip_request(item: dict) -> TripRequest`

- [ ] **Step 1: 拷贝并适配**

参照他 `generate_sft_data.py:459-1165` 的 controlled source 相关函数（`choose_budget_amount`/`build_party_info`/`infer_city_tier`/`build_budget_constraint`/`weighted_choice`/`weighted_block_choice`/`choose_controlled_start_date`/`choose_controlled_accommodation`/`choose_controlled_transportation`/`build_controlled_free_text`/`build_preference_control_spec`/`generate_controlled_request`），搬进 `requests.py`。**不搬** `generate_template_requests` / `generate_llm_request(s)`（是 `--request-source template/llm` 的替代源，2a 只用 controlled）。`control_spec` 至少含 `city_tier/companion_type/budget_level/budget_strictness/pace/diet`。schema import 改我们的包。

- [ ] **Step 2: 写失败测试**

```python
# backend/tests/test_datagen_requests.py
from datetime import date
from app.models.schemas import TripRequest
from ml.planner.datagen.requests import iter_requests, to_trip_request


def test_seed_reproducible():
    a = iter_requests(10, seed=9200, date_mode="mixed")
    b = iter_requests(10, seed=9200, date_mode="mixed")
    assert [x["city"] for x in a] == [x["city"] for x in b]


def test_control_spec_present():
    items = iter_requests(5, seed=9200, date_mode="mixed")
    for it in items:
        cs = it["control_spec"]
        for k in ("city_tier", "companion_type", "budget_level", "budget_strictness"):
            assert k in cs


def test_mixed_mode_includes_past_dates():
    items = iter_requests(40, seed=9200, date_mode="mixed")
    today = date.today()
    assert any(date.fromisoformat(it["end_date"]) < today for it in items)


def test_to_trip_request_valid():
    item = iter_requests(1, seed=9200, date_mode="mixed")[0]
    req = to_trip_request(item)
    assert isinstance(req, TripRequest)
    assert req.travel_days >= 1
```

- [ ] **Step 3: 运行失败 → 对齐他实现的键名/分布 → 通过**

Run: `cd backend && python3 -m pytest tests/test_datagen_requests.py -q -p no:warnings`
Expected: 先 FAIL，对齐后 PASS。若 `control_spec` 字段名不同，按他 `build_preference_control_spec`/`generate_controlled_request` 的实际键名对齐。

- [ ] **Step 4: Commit**

```bash
git add backend/ml/planner/datagen/requests.py backend/tests/test_datagen_requests.py
git commit -m "feat(datagen): controlled request generator with control_spec"
```

---

## Task 4: 泄漏 guard + 拷入他的冻结评测集

**Files:**
- Create: `backend/ml/planner/datagen/leakage.py`
- Overwrite: `backend/ml/planner/eval/records.jsonl`（← 他 200 条）、`backend/ml/planner/eval_hard/records.jsonl`（← 他 300 条）
- Test: `backend/tests/test_datagen_leakage.py`

**Interfaces:**
- Produces:
  - `eval_signature(request: TripRequest) -> str`（字段指纹，与 `ml/planner/requestgen.eval_signature` 同算法，作用于任意 TripRequest）
  - `load_eval_signatures(paths: list[str]) -> set[str]`（读评测记录的 `request` 字段算指纹）

- [ ] **Step 1: 拷入他的评测集**

```bash
cp ~/Desktop/UCLA/helloagents-trip-planner/training/data/planner/eval/records.jsonl \
   backend/ml/planner/eval/records.jsonl
cp ~/Desktop/UCLA/helloagents-trip-planner/training/data/planner/eval_hard/records.jsonl \
   backend/ml/planner/eval_hard/records.jsonl
```
确认行数：`wc -l backend/ml/planner/eval/records.jsonl backend/ml/planner/eval_hard/records.jsonl` → 200 / 300。

- [ ] **Step 2: 写失败测试**

```python
# backend/tests/test_datagen_leakage.py
import json
from app.models.schemas import TripRequest
from ml.planner.datagen.leakage import eval_signature, load_eval_signatures

EVAL = "ml/planner/eval/records.jsonl"


def test_eval_request_is_flagged():
    sigs = load_eval_signatures([EVAL])
    assert len(sigs) == 200
    with open(EVAL, encoding="utf-8") as f:
        first_req = TripRequest(**json.loads(f.readline())["request"])
    assert eval_signature(first_req) in sigs


def test_unrelated_request_not_flagged():
    sigs = load_eval_signatures([EVAL])
    novel = TripRequest(city="张家界", start_date="2019-01-01", end_date="2019-01-02", travel_days=2,
                        transportation="打车", accommodation="经济型酒店", preferences=["摄影"],
                        party={"adults": 1}, budget_constraint={"amount": 1234, "strictness": "soft"})
    assert eval_signature(novel) not in sigs
```

- [ ] **Step 3: 运行失败 → 实现 → 通过**

从 `ml/planner/requestgen.py` 的 `eval_signature` 搬同款算法到 `leakage.py`，`load_eval_signatures` 逐行读记录的 `request` 字段建 `TripRequest` 算指纹。
Run: `cd backend && python3 -m pytest tests/test_datagen_leakage.py -q -p no:warnings`
Expected: 先 FAIL，实现后 PASS。若他 200 条评测请求的 schema 有差异导致 `TripRequest(**...)` 失败，在 `load_eval_signatures` 里对齐字段。

- [ ] **Step 4: Commit**

```bash
git add backend/ml/planner/datagen/leakage.py backend/ml/planner/eval/records.jsonl backend/ml/planner/eval_hard/records.jsonl backend/tests/test_datagen_leakage.py
git commit -m "feat(datagen): leakage guard + adopt helloagents frozen eval sets (200/300)"
```

---

## Task 5: 生成驱动（teacher 循环 + 脚手架 + 记录组装）

**Files:**
- Create: `backend/ml/planner/datagen/generate.py`
- Test: `backend/tests/test_datagen_generate.py`

**Interfaces:**
- Consumes: Task 2 `DataGenPlannerContextBuilder`、Task 3 `iter_requests`/`to_trip_request`、Task 4 `load_eval_signatures`/`eval_signature`、Plan 1 `build_grounded_planner_messages`/`validate_grounded_trip_plan`、`compact_for_planner`
- Produces:
  - `assemble_record(item: dict, context: dict, teacher_output: str) -> dict`（含 `record_id/request/control_spec/planner_context/teacher_output`）
  - `is_clean(plan_json: str, context: dict) -> tuple[bool, list[str]]`（解析 TripPlan → `validate_grounded_trip_plan` → 无违规为干净）
  - `async def main()`（argparse：`--count/--seed/--run-slug/--date-mode/--workers`；产出 `runs/<slug>/{records,requests,errors}.jsonl` + `manifest.json`）

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_datagen_generate.py
import json
from app.models.schemas import TripRequest
from ml.planner.datagen.generate import assemble_record, is_clean

_PLAN = ('{"city":"杭州","start_date":"2020-04-01","end_date":"2020-04-01",'
         '"days":[],"weather_info":[],"overall_suggestions":"ok",'
         '"budget":{"total_attractions":0,"total_hotels":0,"total_meals":0,"total_transportation":0,"total":0}}')


def _ctx():
    return {"request": {"city": "杭州", "start_date": "2020-04-01", "end_date": "2020-04-01"},
            "party": {"total": 2}, "preference_profile": {"diet_avoid": []},
            "planner_constraints": {"expected_dates": ["2020-04-01"]},
            "tool_snapshot": {"trip_weather": [], "classic_pois": [], "preference_pois": [],
                              "scenic_pois": [], "experience_pois": [], "hotel_pois": [], "food_pois": []}}


def test_is_clean_flags_day_count_mismatch():
    # context 期望 1 天，plan 给 0 天 → 有违规 → 不干净
    ok, violations = is_clean(_PLAN, _ctx())
    assert ok is False and violations


def test_assemble_record_shape():
    item = {"city": "杭州", "control_spec": {"budget_level": "standard"}}
    rec = assemble_record(item | {"record_id": "sft_test_0001"}, _ctx(), _PLAN)
    assert rec["record_id"] == "sft_test_0001"
    assert rec["control_spec"]["budget_level"] == "standard"
    assert rec["planner_context"] == _ctx()
    assert json.loads(rec["teacher_output"])["city"] == "杭州"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python3 -m pytest tests/test_datagen_generate.py -q -p no:warnings`
Expected: FAIL（模块/函数不存在）。

- [ ] **Step 3: 实现 generate.py**

参照我们 `ml/planner/data_gen.py` 的脚手架（manifest/resume/usage/单样本容错/asyncio 并发），换三段为 grounded：
- 取数：`DataGenPlannerContextBuilder(amap_key, "open-meteo").collect(req)`（同步，`asyncio.to_thread` 包）；`compact = builder.compact_for_planner(context)`。
- 消息：`teacher.ainvoke(build_grounded_planner_messages(compact))`。
- 清洗：`is_clean()` 解析 `TripPlan(**json.loads(strip_fences(output)))` → `validate_grounded_trip_plan(plan, context)`，无违规为干净；有违规写 `errors.jsonl`。
- 泄漏 guard：`eval_signature(req) in load_eval_signatures(EVAL_PATHS)` 则跳过。
- 记录用 `assemble_record` 组装，只写干净记录到 `records.jsonl`（含 `teacher_output`）。
- teacher = `ChatOpenAI(base_url=deepseek_base_url, api_key=..., model=..., temperature=0.2, max_tokens=8192, timeout=300)`。

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python3 -m pytest tests/test_datagen_generate.py -q -p no:warnings`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/planner/datagen/generate.py backend/tests/test_datagen_generate.py
git commit -m "feat(datagen): teacher generation driver (grounded, scaffolding reused)"
```

---

## Task 6: 导出 sharegpt（train/val）

**Files:**
- Create: `backend/ml/planner/datagen/export.py`
- Test: `backend/tests/test_datagen_export.py`

**Interfaces:**
- Consumes: `runs/<slug>/records.jsonl`（Task 5）、Plan 1 `PLANNER_AGENT_PROMPT`/`build_grounded_planner_messages`/`compact_for_planner`、`validate_grounded_trip_plan`
- Produces:
  - `make_lf_row(record: dict) -> dict`（sharegpt：`{"system": PLANNER_AGENT_PROMPT, "conversations": [{"from":"human","value": <compact context 文本>}, {"from":"gpt","value": teacher_output}]}`，具体键名对齐我们旧 `export_llamafactory.py`）
  - `write_llamafactory_files(records: list[dict], val_ratio: float, out_dir: str) -> tuple[int,int]`（写 `out_dir/train.json`、`out_dir/val.json`，返回 `(train_n, val_n)`）
  - `main()`（argparse：`--runs`（多个 run 目录）`--val-ratio`（默认 0.05）`--out-dir`）

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_datagen_export.py
from app.planner.prompts import PLANNER_AGENT_PROMPT
from ml.planner.datagen.export import make_lf_row, write_llamafactory_files


def _rec(city="杭州"):
    return {"record_id": "r1",
            "planner_context": {"request": {"city": city}, "tool_snapshot": {"food_pois": [{"name": "外婆家"}]}},
            "teacher_output": '{"city":"' + city + '","days":[]}'}


def test_make_lf_row_has_prompt_and_output():
    row = make_lf_row(_rec())
    assert row["system"] == PLANNER_AGENT_PROMPT
    human = row["conversations"][0]["value"]
    gpt = row["conversations"][1]["value"]
    assert "外婆家" in human
    assert '"city":"杭州"' in gpt


def test_split_ratio(tmp_path):
    recs = [_rec(city=f"城{i}") for i in range(20)]
    train_n, val_n = write_llamafactory_files(recs, val_ratio=0.1, out_dir=str(tmp_path))
    assert train_n == 18 and val_n == 2
    assert (tmp_path / "train.json").exists() and (tmp_path / "val.json").exists()
```

- [ ] **Step 2: 运行失败 → 实现 → 通过**

参照旧 `ml/planner/export_llamafactory.py` 的 sharegpt 结构，但 human 侧用 `build_grounded_planner_messages(compact_for_planner(record["planner_context"]))[1].content`，system 用 `PLANNER_AGENT_PROMPT`，gpt 用 `record["teacher_output"]`。`write_llamafactory_files` 按 `val_ratio` 切分（`val_n = round(len*ratio)`）。
Run: `cd backend && python3 -m pytest tests/test_datagen_export.py -q -p no:warnings`
Expected: 先 FAIL，实现后 PASS。

- [ ] **Step 3: Commit**

```bash
git add backend/ml/planner/datagen/export.py backend/tests/test_datagen_export.py
git commit -m "feat(datagen): export grounded records to LLaMA-Factory sharegpt"
```

---

## Task 7: 全量回归 + smoke20 冒烟（真 API）+ 审计

**Files:**（无新增，执行验证）

- [ ] **Step 1: 全量回归**

Run: `cd backend && python3 -m pytest tests/ -q -p no:warnings`
Expected: 全绿（134 旧 + 新增 datagen 测试）。

- [ ] **Step 2: smoke20 冒烟（真高德 + 真 DeepSeek）**

```bash
cd backend && set -a && . ../.env && set +a && \
  nohup caffeinate -is python3 -m ml.planner.datagen.generate \
    --count 20 --seed 9200 --run-slug 260714_smoke20 --date-mode mixed --workers 4 \
    > /tmp/datagen_smoke20.log 2>&1 & disown
```
轮询 `ml/planner/data/runs/260714_smoke20/manifest.json` + `wc -l .../records.jsonl` 直到完成。

- [ ] **Step 3: 人工审计 smoke20**

抽查 `records.jsonl` 若干条：坐标非 0,0、餐厅真实且午≠晚、过去日期的 `trip_weather` 非空（`source=open_meteo_archive`）、预算在约束内。发现系统性问题则回到对应 Task 修复，**不要**直接放大批量。

- [ ] **Step 4: 导出并核对**

```bash
cd backend && python3 -m ml.planner.datagen.export --runs ml/planner/data/runs/260714_smoke20 --val-ratio 0.05 --out-dir ml/planner/llamafactory/generated
```
确认 `train.json`/`val.json` 生成、条数合理、sharegpt 三字段齐全。

- [ ] **Step 5: 提交 smoke 报告（可选）**

审计结论写入 `backend/ml/planner/reports/260714_2a_smoke.md` 并 commit（records/train/val 本身 gitignored）。

> **放大**（100 → ~1200）为执行期操作，非本计划代码任务：改 `--count`/`--run-slug` 重跑，每档人工审计后再放大。目标 ~800-1000 干净样本。

---

## Self-Review 记录

- **Spec 覆盖**：历史天气 → Task 1/2；请求生成器 + control_spec → Task 3；泄漏 guard + 拷评测集 → Task 4；teacher 循环 + 脚手架（manifest/resume/usage/容错）→ Task 5；导出 → Task 6；分阶段跑（smoke→审计→放大）→ Task 7 + 放大说明。
- **类型一致**：`iter_requests`/`to_trip_request`（Task 3）→ Task 5 消费；`DataGenPlannerContextBuilder`（Task 2）→ Task 5；`eval_signature`/`load_eval_signatures`（Task 4）→ Task 5；`assemble_record` 产 `planner_context`/`teacher_output` → Task 6 `make_lf_row` 消费。键名一致。
- **YAGNI 裁剪**：不搬 template/llm request source（Task 3）、不搬他的分布报告脚本（可执行期用 Counter 快查，非必需）、不搬 rerank/DPO（Plan 2b 之后）。
- **超出范围**：LoRA 训练、vLLM serve、三方评测、模型切换 → Plan 2b（需 GPU）。
- **参照非照抄**：移植类步骤给确切源路径 + 行号 + 具体适配点；prompt 逐字（Plan 1 已完成）。
