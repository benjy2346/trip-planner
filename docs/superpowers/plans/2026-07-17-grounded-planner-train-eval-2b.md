# Grounded Planner 训练 + 三方评测（Plan 2b）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Plan 2a 的 grounded 数据 QLoRA 微调 Qwen2.5-7B，在冻结评测集（标准 200 + hard 300）上做 base / 微调 / DeepSeek 三方对比，主指标 `hard_pass`。

**Architecture:** 分两半。**2b-code**（本地 TDD）：把旧的耦合式 `rule_eval.py` 拆成解耦的两个脚本——`generate.py`（记录 + OpenAI 兼容端点 → `generations.jsonl`，只生成不打分）和 `rule_metrics.py`（记录 + `generations.jsonl` → 逐条打分 + 聚合报告），打分复用已就绪的 `validate_grounded_trip_plan` / `recompute_grounded_budget`；再更新 LoRA 配置为 QLoRA + cutoff 24576。**2b-run**（远程 AutoDL 运行手册）：一份带精确命令的 markdown，第一步是 bnb 4-bit 冒烟。

**Tech Stack:** Python 3.12 / pytest / langchain-openai / LLaMA-Factory / vLLM / bitsandbytes（QLoRA）。远程 AutoDL RTX 5090 D 32GB，基础镜像 PyTorch 2.8.0 / CUDA 12.8 / Ubuntu22.04。

## Global Constraints

- **参照 helloagents、写我们自己的**：打分语义对齐 helloagents `training/scripts/eval/eval_rule_metrics.py::evaluate_output`，但逻辑自写、复用我们已移植的 `app/planner/validation.py`。不逐字拷贝其 1070 行实现。
- **train/serve parity**：评测生成的输入必须与训练输入同源——用 `build_grounded_planner_messages(record["compact_planner_context"])`，**不重新 compact**（记录里已烤好 `compact_planner_context`）。system 恒为 `PLANNER_AGENT_PROMPT`（`build_grounded_planner_messages` 已内置）。
- **打分上下文 = `compact_planner_context`**：候选池、party、budget_constraint 都从这里取——评测模型看到的就是它，grounding 要对着模型看到的候选打分。
- **两个评测集独立跑**：`ml/planner/eval/records.jsonl`（标准 200）与 `ml/planner/eval_hard/records.jsonl`（hard 300）分别生成、分别打分、分别出报告；三方对比 = 3 端点 × 2 集 = 6 份报告。
- **主指标 `hard_pass`**（无违规），软指标（grounding 率 / 多样性 / 预算贴合）仅作参考。
- **cutoff_len: 24576**（不是 8192）；**QLoRA 4-bit NF4**（单张 32G 卡装下 24K 上下文）。
- 所有脚本在 `backend/` 目录下用 `python -m ml.planner.eval.<mod>` 运行；测试在 `backend/` 下 `pytest` 运行。

---

## 已核实的现有接口（实现者可直接依赖，无需重新发现）

这些在 Plan 1 已就绪并测试通过，本计划的任务直接调用：

- `app.planner.validation.validate_grounded_trip_plan(plan: TripPlan, context: dict) -> list[str]`
  返回中文违规描述列表，空列表 = 通过。`context` 传 `record["compact_planner_context"]`（含 `request` / `tool_snapshot` / `planner_constraints` / `preference_profile`）。
- `app.planner.validation.recompute_grounded_budget(plan: TripPlan, party_total: int) -> Budget`
  工程重算预算（酒店按 ceil(人数/2) 间、门票餐饮按人数乘）。返回 `app.models.schemas.Budget`，读 `.total`。
- `app.planner.context.build_grounded_planner_messages(context: dict) -> list[BaseMessage]`
  返回 `[SystemMessage(PLANNER_AGENT_PROMPT), HumanMessage("PlannerContext:\n"+json)]`。
- `app.planner.output.name_in_candidates(name: str, candidates: list[str]) -> bool`（模糊别名匹配）
- `app.planner.output.meal_diversity_key(name: str) -> str`（品牌归一，空串=无效名）
- `app.planner.output.is_lodging_breakfast_meal(name: str, meal_type: str) -> bool`
- `app.models.schemas.TripPlan`（Pydantic，`TripPlan(**data)` 解析）

**评测记录字段**（`ml/planner/eval*/records.jsonl` 每行一个 JSON）：
`record_id`（str）、`split`（"eval"）、`compact_planner_context`（dict，含 `request`/`party`/`budget_constraint`/`preference_profile`/`tool_snapshot`/`planner_constraints`）、`planner_context`（完整版，本计划不用）。
标准集 record_id 形如 `planner_standard200_realbudget_eval_000000`；hard 集形如 `planner_hard_realbudget_eval_000000`。
`compact_planner_context["party"]` 形如 `{"total": 2, ...}`；`compact_planner_context["budget_constraint"]` 形如 `{"amount": 9800, "strictness": "hard", ...}`（`amount` 可能为 0/None）。

---

## File Structure

- `backend/ml/planner/eval/__init__.py` — 新建空文件，使 `eval/` 成为可导入包（当前只有数据文件）。
- `backend/ml/planner/eval/generate.py` — 生成 runner：记录 + 端点 → `generations.jsonl`（只 `record_id` + `output`，不打分）。
- `backend/ml/planner/eval/rule_metrics.py` — 打分器：记录 + `generations.jsonl` → 逐条 metrics + 聚合 → `report.json` / `report.md`。
- `backend/tests/test_eval_rule_metrics.py` — rule_metrics 的单测（合成 plan 夹具）。
- `backend/tests/test_eval_generate.py` — generate 的单测（stub LLM，验证 parity + 落盘格式）。
- `backend/ml/planner/configs/qwen25_7b_lora_sft.yaml` — 改为 QLoRA + cutoff 24576。
- `backend/ml/planner/configs/qwen25_7b_lora_merge.yaml` — 新建：把 LoRA adapter 合并成 bf16 完整模型给 vLLM serve。
- `docs/superpowers/runbooks/2026-07-17-grounded-planner-2b-run.md` — 新建：远程 AutoDL 运行手册。

旧的 `backend/ml/planner/rule_eval.py`（非 grounded、耦合式）保留不动——它是这次拆分的参照来源，别删。

---

### Task 1: grounded 打分器 `rule_metrics.py`

把打分逻辑从旧 `rule_eval.py` 迁到 grounded 版：`hard_pass` 复用 `validate_grounded_trip_plan`，预算复用 `recompute_grounded_budget`，再加几个软指标（grounding 率 / 多样性 / 预算贴合）。**meal_scale 显式不做**（YAGNI：需另移植 `is_local_snack_meal` + 档位地板，且不属主对比；需要时再加）。

**Files:**
- Create: `backend/ml/planner/eval/__init__.py`（空文件）
- Create: `backend/ml/planner/eval/rule_metrics.py`
- Test: `backend/tests/test_eval_rule_metrics.py`

**Interfaces:**
- Consumes: `validate_grounded_trip_plan`, `recompute_grounded_budget`（app.planner.validation）；`name_in_candidates`, `meal_diversity_key`, `is_lodging_breakfast_meal`（app.planner.output）；`TripPlan`（app.models.schemas）。
- Produces:
  - `evaluate_output(record: dict, output_text: str) -> dict` — 单条打分，返回含 `record_id`/`json_ok`/`schema_ok`/`hard_pass`/`violations`/`meal_repeat_count`/`attraction_grounding_rate`/`meal_grounding_rate`/`hotel_grounding_rate`/`meal_diversity_unique_rate`/`budget_ok`/`recomputed_total`/`soft_pass` 的 dict。
  - `aggregate(metrics: list[dict]) -> dict` — 聚合成率。
  - `main()` — CLI：`--records` / `--generations` / `--output-dir`。

- [ ] **Step 1: 建包 + 写第一个失败测试（干净 plan 全过）**

先建空包文件：

```bash
touch backend/ml/planner/eval/__init__.py
```

写 `backend/tests/test_eval_rule_metrics.py`。夹具沿用 `backend/tests/test_planner_validation_grounded.py` 的构造风格（同一套 schema），但输出是 JSON 文本（打分器吃字符串）：

```python
"""grounded 打分器单测。夹具与 test_planner_validation_grounded.py 同源：
compact_planner_context 作为打分上下文，plan 序列化成 JSON 文本喂给 evaluate_output。"""
import json

from app.models.schemas import (
    Attraction, Budget, DayPlan, Hotel, Location, Meal, TripPlan,
)
from ml.planner.eval.rule_metrics import aggregate, evaluate_output


def _loc(lng=120.1, lat=30.2):
    return Location(longitude=lng, latitude=lat)


def _ctx(days=2, food=("外婆家", "知味观", "新白鹿", "绿茶餐厅"),
         diet_avoid=(), amount=3000, strictness="soft"):
    dates = [f"2026-08-0{i + 1}" for i in range(days)]
    return {
        "record_id": "planner_standard200_realbudget_eval_000000",
        "split": "eval",
        "compact_planner_context": {
            "request": {"city": "杭州", "start_date": dates[0], "end_date": dates[-1],
                        "travel_days": days, "accommodation": "经济型酒店"},
            "party": {"total": 2},
            "budget_constraint": {"amount": amount, "strictness": strictness},
            "preference_profile": {"diet_avoid": list(diet_avoid)},
            "planner_constraints": {"days_count": days, "expected_dates": dates},
            "tool_snapshot": {
                "trip_weather": [],
                "classic_pois": [{"name": "西湖", "location": {"longitude": 120.1, "latitude": 30.2}}],
                "preference_pois": [{"name": "河坊街", "location": {"longitude": 120.2, "latitude": 30.3}}],
                "scenic_pois": [{"name": "灵隐寺", "location": {"longitude": 120.3, "latitude": 30.4}}],
                "experience_pois": [{"name": "宋城", "location": {"longitude": 120.4, "latitude": 30.5}}],
                "hotel_pois": [{"name": "如家酒店", "location": {"longitude": 120.15, "latitude": 30.25}}],
                "food_pois": [{"name": n, "location": {"longitude": 120.1, "latitude": 30.2}} for n in food],
            },
        },
    }


def _meals(b="酒店早餐", l="知味观", d="新白鹿"):
    return [Meal(type="breakfast", name=b, location=_loc(), estimated_cost=30),
            Meal(type="lunch", name=l, location=_loc(), estimated_cost=60),
            Meal(type="dinner", name=d, location=_loc(), estimated_cost=80)]


def _day(idx, date, attractions=("西湖",), meals=None, hotel="如家酒店"):
    return DayPlan(
        date=date, day_index=idx, description="d", transportation="打车",
        accommodation="经济型酒店",
        hotel=None if hotel is None else Hotel(
            name=hotel, address="x", location=_loc(), distance="", estimated_cost=400),
        attractions=[Attraction(name=n, address="x", location=_loc(), visit_duration=120,
                                description="l", ticket_price=40) for n in attractions],
        meals=meals if meals is not None else _meals())


def _two_day_plan_json(**day2):
    d1 = _day(0, "2026-08-01", attractions=("西湖",), meals=_meals())
    defaults = dict(attractions=("灵隐寺",), meals=_meals(l="绿茶餐厅", d="外婆家"), hotel=None)
    defaults.update(day2)
    d2 = _day(1, "2026-08-02", **defaults)
    plan = TripPlan(city="杭州", start_date="2026-08-01", end_date="2026-08-02",
                    days=[d1, d2], weather_info=[], overall_suggestions="ok",
                    budget=Budget(total_transportation=200))
    return plan.model_dump_json()


def test_clean_plan_hard_passes():
    m = evaluate_output(_ctx(), _two_day_plan_json())
    assert m["json_ok"] and m["schema_ok"]
    assert m["hard_pass"] is True
    assert m["violations"] == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/test_eval_rule_metrics.py::test_clean_plan_hard_passes -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.planner.eval.rule_metrics'`

- [ ] **Step 3: 写 `rule_metrics.py` 最小实现**

```python
"""grounded 规则打分器：读记录 + generations.jsonl → 逐条 metrics + 聚合报告。

与旧 rule_eval.py 的区别：吃 grounded 记录（compact_planner_context）、用
validate_grounded_trip_plan / recompute_grounded_budget，并解耦生成（见 generate.py）。
打分语义参照 helloagents eval_rule_metrics.py::evaluate_output，逻辑复用我们已移植的
app.planner.validation。

运行（在 backend/ 下）：
  python -m ml.planner.eval.rule_metrics \
    --records ml/planner/eval/records.jsonl \
    --generations runs_eval/base_standard/generations.jsonl \
    --output-dir runs_eval/base_standard
"""
import argparse
import json
from pathlib import Path

from app.models.schemas import TripPlan
from app.planner.output import is_lodging_breakfast_meal, meal_diversity_key, name_in_candidates
from app.planner.validation import recompute_grounded_budget, validate_grounded_trip_plan

_ATTRACTION_BUCKETS = ("classic_pois", "preference_pois", "scenic_pois", "experience_pois")


def _strip_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _candidate_names(snapshot: dict, buckets) -> list[str]:
    names = []
    for bucket in buckets:
        for item in snapshot.get(bucket) or []:
            if item.get("name"):
                names.append(item["name"])
    return names


def _rate(hit: int, total: int) -> float:
    return round(hit / total * 100, 1) if total else 100.0


def evaluate_output(record: dict, output_text: str) -> dict:
    ctx = record["compact_planner_context"]
    m = {"record_id": record["record_id"], "json_ok": False, "schema_ok": False,
         "violations": [], "hard_pass": False, "meal_repeat_count": 0,
         "attraction_grounding_rate": 0.0, "meal_grounding_rate": 0.0,
         "hotel_grounding_rate": 0.0, "meal_diversity_unique_rate": 0.0,
         "budget_ok": True, "recomputed_total": None, "soft_pass": False}
    try:
        data = json.loads(_strip_fences(output_text))
        m["json_ok"] = True
        plan = TripPlan(**data)
        m["schema_ok"] = True
    except Exception as e:
        m["violations"] = [f"parse: {e}"]
        return m

    m["violations"] = validate_grounded_trip_plan(plan, ctx)
    m["hard_pass"] = not m["violations"]

    snapshot = ctx["tool_snapshot"]
    attraction_candidates = _candidate_names(snapshot, _ATTRACTION_BUCKETS)
    hotel_candidates = _candidate_names(snapshot, ("hotel_pois",))
    food_candidates = _candidate_names(snapshot, ("food_pois",))

    # grounding 率
    a_hit = a_tot = h_hit = h_tot = f_hit = f_tot = 0
    meal_keys = []
    lunch_dinner_names = []
    for d in plan.days:
        for a in d.attractions:
            a_tot += 1
            if not attraction_candidates or name_in_candidates(a.name, attraction_candidates):
                a_hit += 1
        if d.hotel is not None:
            h_tot += 1
            if not hotel_candidates or name_in_candidates(d.hotel.name, hotel_candidates):
                h_hit += 1
        for meal in d.meals:
            mtype = (meal.type or "").lower()
            if is_lodging_breakfast_meal(meal.name, mtype):
                continue
            f_tot += 1
            if not food_candidates or name_in_candidates(meal.name, food_candidates):
                f_hit += 1
            key = meal_diversity_key(meal.name)
            if key:
                meal_keys.append(key)
            if mtype in ("lunch", "dinner"):
                lunch_dinner_names.append(meal.name)

    m["attraction_grounding_rate"] = _rate(a_hit, a_tot)
    m["meal_grounding_rate"] = _rate(f_hit, f_tot)
    m["hotel_grounding_rate"] = _rate(h_hit, h_tot)
    m["meal_diversity_unique_rate"] = _rate(len(set(meal_keys)), len(meal_keys))
    m["meal_repeat_count"] = len(lunch_dinner_names) - len(set(lunch_dinner_names))

    budget = recompute_grounded_budget(plan, ctx["party"]["total"])
    m["recomputed_total"] = budget.total
    bc = ctx.get("budget_constraint") or {}
    amount = bc.get("amount")
    if amount:
        if bc.get("strictness") == "hard":
            m["budget_ok"] = 0.4 * amount <= budget.total <= amount
        else:
            m["budget_ok"] = budget.total <= 1.2 * amount

    m["soft_pass"] = m["hard_pass"] and m["budget_ok"] and m["meal_repeat_count"] == 0
    return m


def aggregate(metrics: list[dict]) -> dict:
    n = len(metrics)
    rate = lambda k: round(sum(1 for x in metrics if x[k]) / n * 100, 1) if n else 0.0
    avg = lambda k: round(sum(x[k] for x in metrics) / n, 1) if n else 0.0
    return {"count": n, "json_ok": rate("json_ok"), "schema_ok": rate("schema_ok"),
            "hard_pass": rate("hard_pass"), "soft_pass": rate("soft_pass"),
            "budget_ok": rate("budget_ok"),
            "attraction_grounding_avg": avg("attraction_grounding_rate"),
            "meal_grounding_avg": avg("meal_grounding_rate"),
            "hotel_grounding_avg": avg("hotel_grounding_rate"),
            "meal_diversity_unique_avg": avg("meal_diversity_unique_rate"),
            "meal_repeat_avg": round(sum(x["meal_repeat_count"] for x in metrics) / n, 2) if n else 0}


def _render_md(model_tag: str, summary: dict) -> str:
    s = summary
    return (f"# Rule Eval: {model_tag}\n\n| 指标 | 值 |\n| --- | ---: |\n"
            f"| 样本数 | {s['count']} |\n| json_ok | {s['json_ok']}% |\n"
            f"| schema_ok | {s['schema_ok']}% |\n| **hard_pass** | **{s['hard_pass']}%** |\n"
            f"| soft_pass | {s['soft_pass']}% |\n| budget_ok | {s['budget_ok']}% |\n"
            f"| 景点 grounding 均值 | {s['attraction_grounding_avg']}% |\n"
            f"| 餐饮 grounding 均值 | {s['meal_grounding_avg']}% |\n"
            f"| 酒店 grounding 均值 | {s['hotel_grounding_avg']}% |\n"
            f"| 餐饮多样性均值 | {s['meal_diversity_unique_avg']}% |\n"
            f"| 午晚餐平均重复 | {s['meal_repeat_avg']} |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", default="model")
    args = ap.parse_args()

    with open(args.records, encoding="utf-8") as f:
        records = {json.loads(l)["record_id"]: json.loads(l) for l in f if l.strip()}
    metrics = []
    with open(args.generations, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            gen = json.loads(line)
            rec = records.get(gen["record_id"])
            if rec is None:
                continue
            metrics.append(evaluate_output(rec, gen["output"]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in metrics), encoding="utf-8")
    summary = aggregate(metrics)
    (out_dir / "report.json").write_text(
        json.dumps({"model": args.model_tag, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    md = _render_md(args.model_tag, summary)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/test_eval_rule_metrics.py::test_clean_plan_hard_passes -v`
Expected: PASS

- [ ] **Step 5: 补齐关键行为测试**

在测试文件追加：违规被抓、软指标数值正确、budget 硬约束、parse 失败：

```python
def test_ungrounded_attraction_lowers_hard_pass_and_grounding():
    m = evaluate_output(_ctx(), _two_day_plan_json(attractions=("不存在的景点",)))
    assert m["hard_pass"] is False
    assert any("景点" in x and "候选" in x for x in m["violations"])
    # 第2天景点不在候选：2 个景点里命中 1 个
    assert m["attraction_grounding_rate"] == 50.0


def test_meal_grounding_excludes_lodging_breakfast():
    # 酒店早餐不计入 grounding 分母；4 个午晚餐全部命中
    m = evaluate_output(_ctx(), _two_day_plan_json())
    assert m["meal_grounding_rate"] == 100.0


def test_same_day_lunch_dinner_repeat_counted():
    m = evaluate_output(_ctx(), _two_day_plan_json(meals=_meals(l="外婆家", d="外婆家")))
    assert m["meal_repeat_count"] >= 1
    assert m["soft_pass"] is False


def test_hard_budget_overspend_flagged():
    # 硬预算 amount=100，重算 total 远超 → budget_ok False
    m = evaluate_output(_ctx(amount=100, strictness="hard"), _two_day_plan_json())
    assert m["budget_ok"] is False
    assert m["soft_pass"] is False


def test_no_budget_amount_is_ok():
    m = evaluate_output(_ctx(amount=0), _two_day_plan_json())
    assert m["budget_ok"] is True


def test_parse_failure_reported():
    m = evaluate_output(_ctx(), "这不是 JSON")
    assert m["json_ok"] is False
    assert m["hard_pass"] is False
    assert m["violations"] and m["violations"][0].startswith("parse:")


def test_aggregate_reports_rates():
    good = evaluate_output(_ctx(), _two_day_plan_json())
    bad = evaluate_output(_ctx(), "垃圾")
    agg = aggregate([good, bad])
    assert agg["count"] == 2
    assert agg["hard_pass"] == 50.0
    assert agg["json_ok"] == 50.0
```

- [ ] **Step 6: 跑全部测试确认通过**

Run: `cd backend && python -m pytest tests/test_eval_rule_metrics.py -v`
Expected: 全部 PASS（8 个）

- [ ] **Step 7: Commit**

```bash
cd /Users/benjamint/Desktop/UCLA/trip-planner
git add backend/ml/planner/eval/__init__.py backend/ml/planner/eval/rule_metrics.py backend/tests/test_eval_rule_metrics.py
git commit -m "feat(eval): grounded rule_metrics scorer (hard_pass + soft grounding metrics)"
```

---

### Task 2: grounded 生成 runner `generate.py`

从旧 `rule_eval.py` 迁生成半部，改成 grounded：用 `build_grounded_planner_messages(compact_planner_context)`，只写 `generations.jsonl`（`record_id` + `output`），不打分（打分归 rule_metrics）。

**Files:**
- Create: `backend/ml/planner/eval/generate.py`
- Test: `backend/tests/test_eval_generate.py`

**Interfaces:**
- Consumes: `build_grounded_planner_messages`（app.planner.context）；`ChatOpenAI`（langchain_openai）。
- Produces:
  - `messages_for(record: dict) -> list` — 纯函数，返回该记录的 grounded 消息列表（用于 parity 单测）。
  - `write_generations(results: list[tuple[dict, str]], output_dir: str) -> int` — 落盘 `generations.jsonl`，返回写出条数。
  - `main()` — CLI：`--records` / `--base-url` / `--model` / `--api-key-env` / `--output-dir` / `--max-tokens` / `--workers` / `--temperature`。

- [ ] **Step 1: 写失败测试（parity + 落盘）**

`backend/tests/test_eval_generate.py`：

```python
"""generate.py 单测：验证 train/serve parity（用 compact_planner_context + PLANNER_AGENT_PROMPT）
和 generations.jsonl 落盘格式。不打真实网络。"""
import json

from app.planner.prompts import PLANNER_AGENT_PROMPT
from ml.planner.eval.generate import messages_for, write_generations


def _record():
    return {
        "record_id": "r1",
        "compact_planner_context": {"version": "planner-1", "request": {"city": "杭州"}},
    }


def test_messages_use_compact_context_and_planner_prompt():
    msgs = messages_for(_record())
    assert len(msgs) == 2
    assert msgs[0].content == PLANNER_AGENT_PROMPT           # system parity
    assert "PlannerContext:" in msgs[1].content
    assert "杭州" in msgs[1].content                          # compact context 被塞进 human


def test_write_generations_format(tmp_path):
    results = [(_record(), '{"city": "杭州"}')]
    n = write_generations(results, str(tmp_path))
    assert n == 1
    lines = (tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    assert row == {"record_id": "r1", "output": '{"city": "杭州"}'}


def test_write_generations_skips_exceptions(tmp_path):
    results = [(_record(), '{"ok": 1}'), RuntimeError("boom")]
    n = write_generations(results, str(tmp_path))
    assert n == 1  # 异常项被跳过，不落盘
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/test_eval_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.planner.eval.generate'`

- [ ] **Step 3: 写 `generate.py`**

```python
"""grounded 生成 runner：对任意 OpenAI 兼容端点（vLLM /v1 或 DeepSeek）跑冻结评测集，
只写 generations.jsonl（record_id + output），不打分。打分见 rule_metrics.py。

与旧 rule_eval.py 的区别：用 build_grounded_planner_messages(compact_planner_context)
（train/serve parity），且生成与打分解耦（可换端点重跑生成，用同一 rule_metrics 打分）。

运行（在 backend/ 下）：
  # 微调/base 端点（vLLM）
  python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model qwen-ft --api-key-env VLLM_API_KEY \
    --output-dir runs_eval/ft_standard
  # DeepSeek
  python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat \
    --api-key-env DEEPSEEK_API_KEY --output-dir runs_eval/deepseek_standard
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

from app.planner.context import build_grounded_planner_messages


def messages_for(record: dict) -> list:
    """该记录的 grounded 输入消息——直接用记录里已烤好的 compact_planner_context，
    不重新 compact（保证与训练/teacher 见到的输入同源）。"""
    return build_grounded_planner_messages(record["compact_planner_context"])


def write_generations(results: list, output_dir: str) -> int:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            if isinstance(item, Exception):
                print(f"❌ 生成失败: {item}", flush=True)
                continue
            record, text = item
            f.write(json.dumps({"record_id": record["record_id"], "output": text},
                               ensure_ascii=False) + "\n")
            n += 1
    return n


async def _generate(llm: ChatOpenAI, record: dict, sem: asyncio.Semaphore) -> tuple:
    async with sem:
        resp = await llm.ainvoke(messages_for(record))
        return record, resp.content


async def _run(args) -> None:
    with open(args.records, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    llm = ChatOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"),
                     model=args.model, temperature=args.temperature,
                     max_tokens=args.max_tokens, timeout=600)
    sem = asyncio.Semaphore(args.workers)
    results = await asyncio.gather(
        *[_generate(llm, r, sem) for r in records], return_exceptions=True)
    n = write_generations(results, args.output_dir)
    print(f"✅ 写出 {n}/{len(records)} 条 generations 到 {args.output_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/test_eval_generate.py -v`
Expected: 3 个全 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/benjamint/Desktop/UCLA/trip-planner
git add backend/ml/planner/eval/generate.py backend/tests/test_eval_generate.py
git commit -m "feat(eval): grounded generation runner (decoupled generate -> generations.jsonl)"
```

---

### Task 3: QLoRA 训练配置 + 合并导出配置

改 `qwen25_7b_lora_sft.yaml` 为 QLoRA + cutoff 24576；新建合并配置把 adapter 导成 bf16 完整模型给 vLLM。`dataset_info.json` 已正确注册（`columns: {messages: conversations, system: system}`），本任务仅做一个校验测试确认它没被动过。

**Files:**
- Modify: `backend/ml/planner/configs/qwen25_7b_lora_sft.yaml`
- Create: `backend/ml/planner/configs/qwen25_7b_lora_merge.yaml`
- Test: `backend/tests/test_llamafactory_config.py`

**Interfaces:**
- Consumes: `backend/ml/planner/llamafactory/dataset_info.json`（已存在）、`generated/train.json` / `generated/val.json`（已存在，gitignored）。
- Produces: 两个 yaml 配置文件（供 2b-run 在远程用 `llamafactory-cli train/export` 消费）。

- [ ] **Step 1: 写配置校验测试**

`backend/tests/test_llamafactory_config.py`：

```python
"""训练配置与数据集注册的静态校验（不训练，只查关键字段没写错）。"""
import json
from pathlib import Path

import yaml

CFG = Path("ml/planner/configs")
DS = Path("ml/planner/llamafactory/dataset_info.json")


def test_sft_config_is_qlora_long_context():
    cfg = yaml.safe_load((CFG / "qwen25_7b_lora_sft.yaml").read_text())
    assert cfg["cutoff_len"] == 24576                 # 不是 8192
    assert cfg["quantization_bit"] == 4               # QLoRA
    assert cfg["finetuning_type"] == "lora"
    assert cfg["lora_rank"] == 32 and cfg["lora_alpha"] == 64
    assert cfg["dataset"] == "trip_planner_sft"
    assert cfg["eval_dataset"] == "trip_planner_sft_val"
    assert cfg["bf16"] is True                        # 计算精度仍 bf16
    assert cfg["gradient_checkpointing"] is True


def test_merge_config_points_at_adapter():
    cfg = yaml.safe_load((CFG / "qwen25_7b_lora_merge.yaml").read_text())
    assert cfg["adapter_name_or_path"] == cfg_expected_adapter(cfg)
    assert cfg["finetuning_type"] == "lora"
    assert "export_dir" in cfg
    # 合并导出不能再带量化（要导出 bf16 完整模型给 vLLM）
    assert "quantization_bit" not in cfg


def cfg_expected_adapter(cfg):
    # 合并配置的 adapter 路径应等于 sft 配置的 output_dir
    sft = yaml.safe_load((CFG / "qwen25_7b_lora_sft.yaml").read_text())
    return sft["output_dir"]


def test_dataset_info_sharegpt_mapping_intact():
    ds = json.loads(DS.read_text())
    for name in ("trip_planner_sft", "trip_planner_sft_val"):
        entry = ds[name]
        assert entry["formatting"] == "sharegpt"
        assert entry["columns"]["messages"] == "conversations"
        assert entry["columns"]["system"] == "system"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && python -m pytest tests/test_llamafactory_config.py -v`
Expected: FAIL — sft 配置 `cutoff_len` 仍是 8192、无 `quantization_bit`；merge 配置文件不存在。

- [ ] **Step 3: 改 `qwen25_7b_lora_sft.yaml`**

整份替换为：

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all

### QLoRA（4-bit NF4 base，单张 32G 卡装下 24K 上下文）
quantization_bit: 4
quantization_method: bitsandbytes
quantization_type: nf4
double_quantization: true

### dataset
dataset: trip_planner_sft
eval_dataset: trip_planner_sft_val
dataset_dir: ml/planner/llamafactory
template: qwen
cutoff_len: 24576
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: ml/planner/outputs/qwen25_7b_qlora_v1
logging_steps: 5
save_steps: 100
save_total_limit: 4
plot_loss: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
flash_attn: fa2

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
```

> 起点说明（供实现者知情，不必写进文件）：`learning_rate 1e-4`、`num_train_epochs 2` 是 QLoRA 首轮起点，远程看 train/val loss 再调（loss 不降可升 lr 或加到 3 epoch）。`quantization_*` 字段名以远程安装的 LLaMA-Factory 版本为准——2b-run 的 preprocessing dry-run 会校验；若报未知字段，按该版本 `examples/train_qlora/*.yaml` 修正。

- [ ] **Step 4: 新建 `qwen25_7b_lora_merge.yaml`**

```yaml
### 把 QLoRA 训练出的 LoRA adapter 合并进反量化后的 bf16 base，
### 得到完整 bf16 模型给 vLLM serve（vLLM 不吃训练时的 4-bit 格式）。
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
adapter_name_or_path: ml/planner/outputs/qwen25_7b_qlora_v1
template: qwen
finetuning_type: lora
trust_remote_code: true

### export
export_dir: ml/planner/outputs/qwen25_7b_qlora_v1_merged
export_size: 5
export_device: cpu
export_legacy_format: false
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd backend && python -m pytest tests/test_llamafactory_config.py -v`
Expected: 3 个全 PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/benjamint/Desktop/UCLA/trip-planner
git add backend/ml/planner/configs/qwen25_7b_lora_sft.yaml backend/ml/planner/configs/qwen25_7b_lora_merge.yaml backend/tests/test_llamafactory_config.py
git commit -m "feat(train): QLoRA 24k-context sft config + bf16 merge-export config"
```

---

### Task 4: 写远程运行手册

写 2b-run 运行手册（精确命令，bnb 冒烟在最前）。两套评测集（标准 200 + hard 300）**均已 git 跟踪**，远程 `git clone` 会一并拉到，无需额外提交。

**Files:**
- Create: `docs/superpowers/runbooks/2026-07-17-grounded-planner-2b-run.md`

**Interfaces:**
- Consumes: Task 1-3 产出的 `generate.py` / `rule_metrics.py` / 两个 yaml；`generated/train.json`·`val.json`（gitignored，靠 scp 上传）；已跟踪的 `eval/records.jsonl`·`eval_hard/records.jsonl`。
- Produces: 一份可照抄执行的远程手册。

- [ ] **Step 1: 确认两套评测集已跟踪、working tree 干净**

```bash
cd /Users/benjamint/Desktop/UCLA/trip-planner
wc -l backend/ml/planner/eval/records.jsonl backend/ml/planner/eval_hard/records.jsonl
# Expected: 200 和 300
git ls-files --error-unmatch backend/ml/planner/eval_hard/records.jsonl
# Expected: 打印路径（已跟踪）；若报 error 才需要 git add
```

- [ ] **Step 2: 写运行手册**

`docs/superpowers/runbooks/2026-07-17-grounded-planner-2b-run.md`：

````markdown
# Plan 2b 远程运行手册（AutoDL RTX 5090 D）

环境：AutoDL F01 机，RTX 5090 32GB，基础镜像 **PyTorch 2.8.0 / CUDA 12.8 / Ubuntu22.04**，¥2.93/时。
执行方式：本地通过 `ssh`/`scp` 驱动；长任务（训练）在远程 `nohup ... &` 后台跑 + 轮询。
**用不即关机**（含调试卡住时）。全流程约 6–12 小时、¥20–40。

## 0. 连接（一次性）
- 推荐 SSH key 免密：把本地公钥加进 AutoDL 实例 `~/.ssh/authorized_keys`，或配 `~/.ssh/config` 别名 `autodl`。
- 之后所有步骤都能 `ssh autodl 'cmd'` 非交互执行。

## 1. ⚠️ bnb 4-bit 冒烟（开机第一件事，别跳过）
QLoRA 依赖 bitsandbytes，其对 Blackwell(sm_120) 的支持是近期才有的。**先验证再训练**：
```bash
pip install -U bitsandbytes
python - <<'PY'
import torch, transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
print("cuda ok:", torch.zeros(1).cuda().is_cuda, "| cap:", torch.cuda.get_device_capability())
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb, device_map="cuda", trust_remote_code=True)
print("4-bit load OK, layers:", m.config.num_hidden_layers)
PY
```
- **通过** → 继续 QLoRA。
- **失败**（bnb 报 sm_120 不支持）→ 退路二选一：① sft 配置去掉 `quantization_*`、`cutoff_len` 降到能装下的 bf16 单卡值（先试 16384）；② 加租第二张卡走 bf16 + Ulysses 序列并行。**别硬扛，先决定路线再往下。**

## 2. 传输 + 环境
```bash
# 远程 clone 本仓库（含代码 + committed 的 500 条评测记录）
ssh autodl 'cd ~ && git clone <repo-url> trip-planner && cd trip-planner && git checkout feat/planner-finetune'
# 上传 gitignored 的训练数据
scp backend/ml/planner/llamafactory/generated/train.json autodl:~/trip-planner/backend/ml/planner/llamafactory/generated/
scp backend/ml/planner/llamafactory/generated/val.json   autodl:~/trip-planner/backend/ml/planner/llamafactory/generated/
# 装依赖
ssh autodl 'pip install -U "llamafactory[torch,metrics,bitsandbytes]" vllm langchain-openai'
```
下模型（ModelScope，国内快）：
```bash
ssh autodl 'pip install -U modelscope && modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ~/models/Qwen2.5-7B-Instruct'
```
> 若用本地已下模型，把两个 yaml 里的 `model_name_or_path` 指到 `~/models/Qwen2.5-7B-Instruct`。

## 3. 预处理 dry-run（先验证数据集加载 + 截断比例）
```bash
ssh autodl 'cd ~/trip-planner/backend && llamafactory-cli train ml/planner/configs/qwen25_7b_lora_sft.yaml \
  --max_steps 1 --output_dir /tmp/dryrun'
```
- 确认能加载 `trip_planner_sft` / `_val`、sharegpt 映射无报错、截断比例合理（>24576 的样本约 10–20%）。
- 若报 `quantization_*` 未知字段，按该 LLaMA-Factory 版本 `examples/train_qlora/*.yaml` 修正字段名，改完重跑本步。

## 4. 训练（后台 + 轮询）
```bash
ssh autodl 'cd ~/trip-planner/backend && nohup llamafactory-cli train ml/planner/configs/qwen25_7b_lora_sft.yaml \
  > ~/train.log 2>&1 &'
# 轮询
ssh autodl 'tail -n 30 ~/train.log; nvidia-smi --query-gpu=memory.used --format=csv'
```
- 看 train/val loss 下降。若 OOM：先把 `cutoff_len` 降到 16384 重训（仍远好于 8192）。
- 产出 LoRA adapter 在 `ml/planner/outputs/qwen25_7b_qlora_v1`。

## 5. 合并成 bf16 完整模型（给 vLLM）
```bash
ssh autodl 'cd ~/trip-planner/backend && llamafactory-cli export ml/planner/configs/qwen25_7b_lora_merge.yaml'
# 产出：ml/planner/outputs/qwen25_7b_qlora_v1_merged
```

## 6. 三方生成（3 端点 × 2 评测集 = 6 份 generations）
每个 vLLM 端点**必须** `--max-model-len 32768`（输入中位 ~22.5K、最大 ~31.5K token）。
```bash
# --- base 7B ---
ssh autodl 'cd ~/trip-planner/backend && nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Qwen2.5-7B-Instruct --max-model-len 32768 --port 8000 > ~/vllm_base.log 2>&1 &'
# 起来后（tail 日志看到 "Uvicorn running"）：
ssh autodl 'cd ~/trip-planner/backend && \
  python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model base --api-key-env NONE --output-dir runs_eval/base_standard && \
  python -m ml.planner.eval.generate --records ml/planner/eval_hard/records.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model base --api-key-env NONE --output-dir runs_eval/base_hard'
# 停 base vLLM，再起 merged（换 --model 路径，同 --port 8000），跑 ft_standard / ft_hard：
#   --model ~/trip-planner/backend/ml/planner/outputs/qwen25_7b_qlora_v1_merged --output-dir runs_eval/ft_{standard,hard}
# --- DeepSeek（API，无需 GPU）---
ssh autodl 'cd ~/trip-planner/backend && export DEEPSEEK_API_KEY=<key> && \
  python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat --output-dir runs_eval/deepseek_standard && \
  python -m ml.planner.eval.generate --records ml/planner/eval_hard/records.jsonl \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat --output-dir runs_eval/deepseek_hard'
```

## 7. 打分（6 份 generations → 6 份报告）
```bash
ssh autodl 'cd ~/trip-planner/backend && for tag in base ft deepseek; do \
  python -m ml.planner.eval.rule_metrics --records ml/planner/eval/records.jsonl \
    --generations runs_eval/${tag}_standard/generations.jsonl --output-dir runs_eval/${tag}_standard --model-tag ${tag}_standard; \
  python -m ml.planner.eval.rule_metrics --records ml/planner/eval_hard/records.jsonl \
    --generations runs_eval/${tag}_hard/generations.jsonl --output-dir runs_eval/${tag}_hard --model-tag ${tag}_hard; \
done'
```
成功判据：微调 7B 的 `hard_pass` 显著高于 base 7B，并向 DeepSeek 靠拢。

## 8. 回收 + 关机
```bash
scp -r autodl:~/trip-planner/backend/runs_eval ./backend/ml/planner/reports_2b/     # 拉回 6 份报告
scp -r autodl:~/trip-planner/backend/ml/planner/outputs/qwen25_7b_qlora_v1 ./         # 拉回 adapter（可选）
# 确认无残留后台任务后，AutoDL 控制台关机止损。
```
本地把 6 份报告整理成一份三方对比，committed 到 `backend/ml/planner/reports_2b/`。
````

- [ ] **Step 3: Commit**

```bash
cd /Users/benjamint/Desktop/UCLA/trip-planner
git add docs/superpowers/runbooks/2026-07-17-grounded-planner-2b-run.md
git commit -m "docs(2b): remote AutoDL run runbook"
```

---

## Self-Review 结论

- **Spec 覆盖**：2b-code 三件（rule_metrics.py=Task1 / generate.py=Task2 / dataset 注册+配置=Task3）全覆盖；2b-run 运行手册=Task4，bnb 冒烟为第一步 ✓。三方评测（base/ft/deepseek）、`--max-model-len 32768`、cutoff 24576、QLoRA、合并 bf16 serve、成本/关机——均落到具体步骤 ✓。
- **对 spec 的一处显式收窄**：`meal_scale` 软指标不做（YAGNI，需另移植 `is_local_snack_meal` + 档位地板，且不属主对比）。软指标保留 grounding 率 / 多样性 / 预算贴合，覆盖 spec「指标」节的 grounding/多样性/预算贴合率。
- **评测集 git 状态**：标准 200 + hard 300 均已跟踪（早先误判 hard 未跟踪，已核实纠正），远程 clone 直接拿到；Task4 仅做一次确认，不再提交。
- **Type/接口一致性**：`evaluate_output(record, output_text)` 在 rule_metrics 定义、被其 `main()` 调用；`messages_for`/`write_generations` 在 generate 定义、被单测与 `main()` 调用；两 yaml 的 `output_dir`/`adapter_name_or_path` 对应关系由 Task3 测试锁定 ✓。
- **parity**：generate 用 `build_grounded_planner_messages(record["compact_planner_context"])`，system=PLANNER_AGENT_PROMPT，与训练 `make_lf_row` 同源 ✓。
