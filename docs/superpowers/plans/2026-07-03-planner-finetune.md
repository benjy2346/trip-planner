# 行程生成模型微调（Planner SFT）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 参照 helloagents-trip-planner 的后训练主线，为 `assembler_node` 训练一个 LoRA 微调的 Qwen2.5-7B 行程生成模型：先做协议改造与冻结评测，再用 DeepSeek 做 teacher 生成审计过的 SFT 数据，LLaMA-Factory 训练后经 vLLM 接入 `llm_router`，DeepSeek 兜底常驻。

**Architecture:** `app/planner/` 下 `context.py`（PlannerContext + prompt，训练/评测/推理单一来源）与 `validation.py`（shape 校验 + 预算重算）；`ml/planner/` 下请求生成、评测集构建、规则评测、teacher 数据生成、LLaMA-Factory 导出；接入走 `agents_config.yaml` + `get_agent_llm` 现有 provider 机制。

**Tech Stack:** Python 3.11 / LangGraph / FastAPI / LLaMA-Factory / vLLM / Qwen2.5-7B-Instruct

## Global Constraints

- Python 3.11；backend **不新增运行时依赖**（训练在独立 LLaMA-Factory 环境；推理经 HTTP 调 vLLM）。
- 现有测试保持全绿（当前 36 个；若意图微调计划已落地则为其全量）。
- 冻结评测集 `ml/planner/eval/records.jsonl`（standard 50）与 `ml/planner/eval_hard/records.jsonl`（hard 50）入 git；**建成后不再重采样请求**。
- 训练数据与评测集零重叠：数据生成必须用 `eval_signature()` 过滤，manifest 记录 `eval_overlap_skipped`。
- 每个数据 run 写入独立目录 `backend/ml/planner/data/runs/<YYMMDD>_<slug>/`，含 `requests.jsonl / records.jsonl / errors.jsonl / manifest.json`（manifest 必须含 token usage）。
- 数据 run、LLaMA-Factory 导出产物、训练输出不入 git；评测集、脚本、configs、轻量报告入 git。
- 训练口径：Qwen2.5-7B-Instruct、LoRA r=32 / alpha=64 / dropout 0.05 / target all、cutoff_len=8192、lr 5e-5、3 epoch、bf16。
- 成功标准：SFT 模型 hardpass standard ≥ 85%、hard ≥ 70%、比未微调基座高 ≥ 20pp、不低于 DeepSeek 基线 −5pp；达标前不切 `agents_config.yaml`。
- `hotel.distance` 必须为空字符串；prompt/数据/校验中一律禁止伪距离与占位餐饮词。

---

## File Structure

```
backend/
  app/
    models/schemas.py          # 改造：+PartyInfo, +BudgetConstraint, TripRequest 接入（可选带默认）
    planner/
      __init__.py              # 新增（空）
      context.py               # 新增：PlannerContext 构建 + Planner prompt（单一来源）
      validation.py            # 新增：TripPlan shape 校验 + recompute_budget
    agents/
      supervisor.py            # 改造：assembler_node 用 context/validation/acall_agent_with_fallback
      llm_router.py            # 改造：+acall_agent_with_fallback
    config.py                  # 改造：+local_base_url / local_api_key / local_model
  agents_config.yaml           # 改造（Task 10 达标后）：assembler → local
  ml/planner/
    requestgen.py              # 新增：受控请求生成 + eval_signature
    build_eval_set.py          # 新增：跑子图拍快照，构建冻结评测集
    rule_eval.py               # 新增：hardpass/softpass 规则评测 CLI
    data_gen.py                # 新增：teacher 数据生成（run 目录 + manifest + usage）
    export_llamafactory.py     # 新增：sharegpt 导出 + train/val 切分
    configs/qwen25_7b_lora_sft.yaml   # 新增：LLaMA-Factory 训练配置
    llamafactory/dataset_info.json    # 新增（generated/ 不入 git）
    eval/records.jsonl         # 生成后入 git（冻结）
    eval_hard/records.jsonl    # 生成后入 git（冻结）
    reports/                   # 轻量评测报告（md，入 git）
  tests/
    test_planner_schemas.py    # 新增
    test_planner_context.py    # 新增
    test_planner_validation.py # 新增
    test_requestgen.py         # 新增
    test_rule_eval.py          # 新增
    test_export_llamafactory.py# 新增
    test_supervisor.py         # 更新（patch 目标改为 acall_agent_with_fallback）
  .gitignore                   # 追加 ml/planner 产物目录
```

---

## Task 1: 结构化 party 与预算约束（schemas）

**Files:**
- Modify: `backend/app/models/schemas.py`
- Modify: `backend/.gitignore`
- Test: `backend/tests/test_planner_schemas.py`

**Interfaces:**
- Produces:
  - `class PartyInfo(BaseModel)`：`adults:int=1, children:int=0, elders:int=0`，computed 属性 `total:int`
  - `class BudgetConstraint(BaseModel)`：`amount:Optional[int]=None, scope:str="total", currency:str="CNY", budget_level:Literal["limited","comfortable","premium"]="comfortable", strictness:Literal["hard","soft"]="soft"`
  - `TripRequest` 新字段：`party: PartyInfo`（default_factory）、`budget_constraint: Optional[BudgetConstraint] = None`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_planner_schemas.py
import pytest
from pydantic import ValidationError
from app.models.schemas import TripRequest, PartyInfo, BudgetConstraint


def _base_kwargs():
    return dict(
        city="北京", start_date="2026-08-01", end_date="2026-08-03",
        travel_days=3, transportation="公共交通", accommodation="经济型酒店",
    )


def test_trip_request_backward_compatible_defaults():
    req = TripRequest(**_base_kwargs())
    assert req.party.adults == 1
    assert req.party.total == 1
    assert req.budget_constraint is None


def test_party_total_computed():
    p = PartyInfo(adults=2, children=1, elders=1)
    assert p.total == 4
    assert p.model_dump()["total"] == 4


def test_party_rejects_zero_adults():
    with pytest.raises(ValidationError):
        PartyInfo(adults=0)


def test_budget_constraint_fields():
    b = BudgetConstraint(amount=3500, budget_level="limited", strictness="hard")
    assert b.scope == "total"
    assert b.currency == "CNY"


def test_budget_rejects_bad_level():
    with pytest.raises(ValidationError):
        BudgetConstraint(budget_level="luxury")


def test_trip_request_accepts_structured_fields():
    req = TripRequest(
        **_base_kwargs(),
        party={"adults": 2, "children": 1},
        budget_constraint={"amount": 5000, "strictness": "hard"},
    )
    assert req.party.total == 3
    assert req.budget_constraint.amount == 5000
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_planner_schemas.py -q`
Expected: FAIL with `ImportError: cannot import name 'PartyInfo'`

- [ ] **Step 3: 修改 schemas.py**

在 `backend/app/models/schemas.py` 顶部 import 段将 `from typing import List, Optional, Union` 改为：

```python
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, computed_field
```

在 `# ============ 请求模型 ============` 之后、`class TripRequest` 之前新增：

```python
class PartyInfo(BaseModel):
    """同行人信息"""
    adults: int = Field(default=1, ge=1, description="成人数")
    children: int = Field(default=0, ge=0, description="儿童数")
    elders: int = Field(default=0, ge=0, description="老人数")

    @computed_field
    @property
    def total(self) -> int:
        return self.adults + self.children + self.elders


class BudgetConstraint(BaseModel):
    """预算约束"""
    amount: Optional[int] = Field(default=None, ge=0, description="预算金额(元)，None 表示未指定")
    scope: str = Field(default="total", description="预算口径：整趟总额")
    currency: str = Field(default="CNY", description="币种")
    budget_level: Literal["limited", "comfortable", "premium"] = Field(
        default="comfortable", description="预算档位")
    strictness: Literal["hard", "soft"] = Field(
        default="soft", description="hard 不能超，soft 尽量贴合")
```

在 `TripRequest` 的 `free_text_input` 字段之后新增两个字段：

```python
    party: PartyInfo = Field(default_factory=PartyInfo, description="同行人信息")
    budget_constraint: Optional[BudgetConstraint] = Field(default=None, description="预算约束")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_planner_schemas.py tests/ -q`
Expected: 新增 6 个通过，且现有测试全绿（新字段均有默认值，不破坏旧构造）。

- [ ] **Step 5: 更新 .gitignore**

在 `backend/.gitignore` 末尾追加：

```
# Planner 微调产物（评测集与 configs 入 git，数据/训练产物不入）
ml/planner/data/
ml/planner/outputs/
ml/planner/llamafactory/generated/
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/models/schemas.py backend/tests/test_planner_schemas.py backend/.gitignore
git commit -m "feat: add PartyInfo and BudgetConstraint to TripRequest for planner finetune"
```

---

## Task 2: PlannerContext 构建器与 prompt（单一来源）

**Files:**
- Create: `backend/app/planner/__init__.py`（空文件）
- Create: `backend/app/planner/context.py`
- Test: `backend/tests/test_planner_context.py`

**Interfaces:**
- Consumes: `TripRequest / PartyInfo / BudgetConstraint`（Task 1）；`WeatherInfo / Hotel / Attraction`（现有）
- Produces:
  - `build_planner_context(request, weather_outputs, hotel_outputs, poi_outputs) -> dict`（键：request/party/budget_constraint/lodging_policy/pricing_policy/tool_snapshot/planner_constraints）
  - `build_planner_messages(context: dict) -> list[BaseMessage]`
  - `PLANNER_SYSTEM_PROMPT: str`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_planner_context.py
from app.models.schemas import TripRequest, WeatherInfo, Hotel, Attraction, Location
from app.planner.context import (
    build_planner_context, build_planner_messages, PLANNER_SYSTEM_PROMPT,
)


def _req():
    return TripRequest(
        city="杭州", start_date="2026-08-01", end_date="2026-08-03",
        travel_days=3, transportation="打车", accommodation="经济型酒店",
        preferences=["美食"], free_text_input="预算3500左右",
        party={"adults": 2}, budget_constraint={"amount": 3500, "strictness": "hard"},
    )


def _snapshot_inputs():
    weather = [WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云",
                           day_temp=30, night_temp=22)]
    hotels = [Hotel(name="如家杭州店", address="上城区", estimated_cost=250, type="经济型")]
    pois = [Attraction(name="西湖", address="西湖区", location=Location(longitude=120.1, latitude=30.2),
                       visit_duration=120, description="湖", ticket_price=0)]
    return weather, hotels, pois


def test_context_has_all_sections():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    assert set(ctx.keys()) == {
        "request", "party", "budget_constraint", "lodging_policy",
        "pricing_policy", "tool_snapshot", "planner_constraints",
    }


def test_dates_and_lodging():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    assert ctx["planner_constraints"]["dates"] == ["2026-08-01", "2026-08-02", "2026-08-03"]
    assert ctx["lodging_policy"]["nights"] == 2
    assert ctx["lodging_policy"]["hotel_on_last_day"] is False


def test_party_and_budget_compiled():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    assert ctx["party"]["total"] == 2
    assert ctx["budget_constraint"]["amount"] == 3500
    assert ctx["budget_constraint"]["strictness"] == "hard"


def test_default_budget_when_absent():
    req = _req()
    req.budget_constraint = None
    ctx = build_planner_context(req, *_snapshot_inputs())
    assert ctx["budget_constraint"]["amount"] is None
    assert ctx["budget_constraint"]["strictness"] == "soft"


def test_snapshot_counts():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    counts = ctx["tool_snapshot"]["candidate_counts"]
    assert counts == {"weather": 1, "hotels": 1, "attractions": 1}


def test_prompt_bans_fake_distance_and_placeholders():
    assert "距离景点2公里" not in PLANNER_SYSTEM_PROMPT
    assert "占位" in PLANNER_SYSTEM_PROMPT
    assert '"distance": ""' in PLANNER_SYSTEM_PROMPT


def test_messages_carry_context_json():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    msgs = build_planner_messages(ctx)
    assert len(msgs) == 2
    assert msgs[0].content == PLANNER_SYSTEM_PROMPT
    assert "西湖" in msgs[1].content and "如家杭州店" in msgs[1].content
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_planner_context.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.planner'`

- [ ] **Step 3: 创建 context.py（先 `touch backend/app/planner/__init__.py`）**

```python
# backend/app/planner/context.py
"""PlannerContext：把请求与子图结构化结果编译成模型可见的开卷资料。

训练数据生成、规则评测、线上推理共用此单一来源，保证三处输入协议一致。
"""
import json
from datetime import date, timedelta
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from app.models.schemas import TripRequest, WeatherInfo, Hotel, Attraction

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
5. 除最后一天外每天 hotel 不能为 null，整个行程连续入住同一家酒店（从 tool_snapshot.hotel_candidates 中选取）；最后一天 hotel 为 null。
6. hotel.distance 必须为空字符串 ""，没有路线工具时不得编造距离。
7. 每天必须包含 breakfast/lunch/dinner 三餐（最后一天也不能缺晚餐），餐饮必须写具体店名，禁止"早餐推荐""附近餐厅""当地小吃""酒店晚餐""无"这类占位词。
8. 价格口径见 pricing_policy：酒店按单间每晚，门票按成人单人票，餐饮按单人单餐。budget 分项 = 单价 × 对应数量（门票和餐饮要乘 party.total 人数，酒店乘住宿晚数），total 为各分项之和。
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
        "pricing_policy": PRICING_POLICY,
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_planner_context.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/planner/ backend/tests/test_planner_context.py
git commit -m "feat: add PlannerContext builder and planner prompt (single source)"
```

---

## Task 3: TripPlan 校验与预算重算

**Files:**
- Create: `backend/app/planner/validation.py`
- Test: `backend/tests/test_planner_validation.py`

**Interfaces:**
- Consumes: `TripPlan / Budget`（现有）；context dict（Task 2 结构）
- Produces:
  - `validate_trip_plan(plan: TripPlan, context: dict) -> list[str]`（violations，空列表 = hardpass）
  - `recompute_budget(plan: TripPlan, party_total: int) -> Budget`
  - `MEAL_PLACEHOLDER_RE`（正则，供 rule_eval 复用）

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_planner_validation.py
from app.models.schemas import (
    TripRequest, TripPlan, DayPlan, Attraction, Meal, Hotel, WeatherInfo, Location, Budget,
)
from app.planner.context import build_planner_context
from app.planner.validation import validate_trip_plan, recompute_budget


def _ctx():
    req = TripRequest(
        city="杭州", start_date="2026-08-01", end_date="2026-08-02",
        travel_days=2, transportation="打车", accommodation="经济型酒店",
        party={"adults": 2},
    )
    weather = [
        WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云", day_temp=30, night_temp=22),
        WeatherInfo(date="2026-08-02", day_weather="多云", night_weather="多云", day_temp=31, night_temp=23),
    ]
    hotels = [Hotel(name="如家杭州店", address="上城区", estimated_cost=250, type="经济型")]
    pois = [
        Attraction(name="西湖", address="西湖区", location=Location(longitude=120.1, latitude=30.2),
                   visit_duration=120, description="湖", ticket_price=0),
        Attraction(name="灵隐寺", address="西湖区", location=Location(longitude=120.10, latitude=30.24),
                   visit_duration=120, description="寺", ticket_price=30),
    ]
    return build_planner_context(req, weather, hotels, pois)


def _hotel():
    return Hotel(name="如家杭州店", address="上城区", estimated_cost=250, type="经济型", distance="")


def _meals(prefix):
    return [
        Meal(type="breakfast", name=f"{prefix}豆浆店", estimated_cost=15),
        Meal(type="lunch", name=f"{prefix}面馆", estimated_cost=40),
        Meal(type="dinner", name=f"{prefix}杭帮菜", estimated_cost=80),
    ]


def _good_plan():
    return TripPlan(
        city="杭州", start_date="2026-08-01", end_date="2026-08-02",
        days=[
            DayPlan(date="2026-08-01", day_index=0, description="d1", transportation="打车",
                    accommodation="经济型酒店", hotel=_hotel(),
                    attractions=[Attraction(name="西湖", address="西湖区",
                                            location=Location(longitude=120.1, latitude=30.2),
                                            visit_duration=120, description="湖", ticket_price=0)],
                    meals=_meals("知味观")),
            DayPlan(date="2026-08-02", day_index=1, description="d2", transportation="打车",
                    accommodation="经济型酒店", hotel=None,
                    attractions=[Attraction(name="灵隐寺", address="西湖区",
                                            location=Location(longitude=120.10, latitude=30.24),
                                            visit_duration=120, description="寺", ticket_price=30)],
                    meals=_meals("外婆家")),
        ],
        weather_info=[
            WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云", day_temp=30, night_temp=22),
            WeatherInfo(date="2026-08-02", day_weather="多云", night_weather="多云", day_temp=31, night_temp=23),
        ],
        overall_suggestions="ok",
        budget=Budget(total_transportation=200),
    )


def test_good_plan_passes():
    assert validate_trip_plan(_good_plan(), _ctx()) == []


def test_wrong_day_count_flagged():
    plan = _good_plan()
    plan.days = plan.days[:1]
    assert any("days" in v for v in validate_trip_plan(plan, _ctx()))


def test_missing_dinner_flagged():
    plan = _good_plan()
    plan.days[1].meals = plan.days[1].meals[:2]  # 去掉最后一天 dinner
    assert any("dinner" in v for v in validate_trip_plan(plan, _ctx()))


def test_placeholder_meal_flagged():
    plan = _good_plan()
    plan.days[0].meals[1].name = "附近餐厅"
    assert any("占位" in v for v in validate_trip_plan(plan, _ctx()))


def test_missing_hotel_on_lodging_day_flagged():
    plan = _good_plan()
    plan.days[0].hotel = None
    assert any("hotel" in v for v in validate_trip_plan(plan, _ctx()))


def test_fake_distance_flagged():
    plan = _good_plan()
    plan.days[0].hotel.distance = "距离景点2公里"
    assert any("distance" in v for v in validate_trip_plan(plan, _ctx()))


def test_ungrounded_attraction_flagged():
    plan = _good_plan()
    plan.days[0].attractions[0].name = "编造乐园"
    assert any("候选" in v for v in validate_trip_plan(plan, _ctx()))


def test_weather_mismatch_flagged():
    plan = _good_plan()
    plan.weather_info[0].day_weather = "暴雪"
    assert any("天气" in v for v in validate_trip_plan(plan, _ctx()))


def test_recompute_budget_units():
    b = recompute_budget(_good_plan(), party_total=2)
    assert b.total_hotels == 250            # 1 晚 × 250（按有 hotel 的天数）
    assert b.total_attractions == 60        # (0+30) × 2 人
    assert b.total_meals == 540             # (15+40+80)×2天 ×2人
    assert b.total_transportation == 200    # 沿用模型自报
    assert b.total == 250 + 60 + 540 + 200
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_planner_validation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.planner.validation'`

- [ ] **Step 3: 创建 validation.py**

```python
# backend/app/planner/validation.py
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_planner_validation.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/planner/validation.py backend/tests/test_planner_validation.py
git commit -m "feat: add TripPlan shape validation and engineering budget recompute"
```

---

## Task 4: assembler 接入 PlannerContext 与按 agent 路由的 LLM 调用

**Files:**
- Modify: `backend/app/agents/llm_router.py`
- Modify: `backend/app/agents/supervisor.py`
- Test: `backend/tests/test_llm_router_agent.py`（追加）、`backend/tests/test_supervisor.py`（更新 patch 目标）

**Interfaces:**
- Consumes: `build_planner_context / build_planner_messages`（Task 2）、`validate_trip_plan / recompute_budget`（Task 3）、`get_agent_llm / acall_with_fallback`（现有）
- Produces: `async acall_agent_with_fallback(agent_name: str, messages: list[BaseMessage])` —— 优先用 `agents_config.yaml` 中该 agent 的模型，任何异常降级到全局 DeepSeek 链。**Task 10 的本地模型接入只改配置，不再改代码。**

- [ ] **Step 1: 写失败测试（追加到 test_llm_router_agent.py 末尾）**

```python
@pytest.mark.asyncio
async def test_acall_agent_with_fallback_uses_agent_llm():
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.agents.llm_router import acall_agent_with_fallback

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value="agent-reply")
    with patch("app.agents.llm_router.get_agent_llm", return_value=mock_llm), \
         patch("app.agents.llm_router.acall_with_fallback", AsyncMock()) as mock_global:
        result = await acall_agent_with_fallback("assembler", ["msg"])
    assert result == "agent-reply"
    mock_global.assert_not_called()


@pytest.mark.asyncio
async def test_acall_agent_with_fallback_degrades_to_global_chain():
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.agents.llm_router import acall_agent_with_fallback

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError("down"))
    with patch("app.agents.llm_router.get_agent_llm", return_value=mock_llm), \
         patch("app.agents.llm_router.acall_with_fallback",
               AsyncMock(return_value="global-reply")) as mock_global:
        result = await acall_agent_with_fallback("assembler", ["msg"])
    assert result == "global-reply"
    mock_global.assert_called_once()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_llm_router_agent.py -q`
Expected: FAIL with `ImportError: cannot import name 'acall_agent_with_fallback'`

- [ ] **Step 3: 在 llm_router.py 末尾追加**

```python
async def acall_agent_with_fallback(agent_name: str, messages: list[BaseMessage]):
    """优先使用 agents_config.yaml 中该 agent 指定的模型，异常时降级到全局链。"""
    try:
        return await get_agent_llm(agent_name).ainvoke(messages)
    except Exception:
        return await acall_with_fallback(messages)
```

- [ ] **Step 4: 改造 supervisor.py 的 assembler_node**

import 段追加：

```python
from app.agents.llm_router import acall_with_fallback, acall_agent_with_fallback
from app.planner.context import build_planner_context, build_planner_messages
from app.planner.validation import validate_trip_plan, recompute_budget
```

将 `assembler_node`（原第 61-157 行，含内嵌 `PLANNER_SYSTEM_PROMPT`）整体替换为：

```python
async def assembler_node(state: SupervisorState) -> dict:
    req = state["trip_request"]
    context = build_planner_context(
        req,
        state.get("weather_outputs", []),
        state.get("hotel_outputs", []),
        state.get("poi_outputs", []),
    )
    response = await acall_agent_with_fallback("assembler", build_planner_messages(context))
    content = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    trip_plan = TripPlan(**json.loads(content))

    violations = validate_trip_plan(trip_plan, context)
    if violations:
        print(f"⚠️ TripPlan 校验告警 {len(violations)} 条: {violations[:5]}")
    trip_plan.budget = recompute_budget(trip_plan, context["party"]["total"])

    return {
        "trip_plan": trip_plan,
        "messages": [AIMessage(content=f"已为您生成{req.city}{req.travel_days}天行程。")],
    }
```

- [ ] **Step 5: 更新 test_supervisor.py 的 patch 目标**

两个测试中的

```python
         patch("app.agents.supervisor.acall_with_fallback", AsyncMock(return_value=mock_response)):
```

均改为

```python
         patch("app.agents.supervisor.acall_agent_with_fallback", AsyncMock(return_value=mock_response)):
```

（`days=[]` 的极简 plan 会触发校验告警日志，但只告警不拦截，断言不受影响。）

- [ ] **Step 6: 全量回归**

Run: `cd backend && python3 -m pytest tests/ -q`
Expected: 全绿（含更新后的 supervisor 测试与新增 router 测试）。

- [ ] **Step 7: Commit**

```bash
git add backend/app/agents/llm_router.py backend/app/agents/supervisor.py \
        backend/tests/test_llm_router_agent.py backend/tests/test_supervisor.py
git commit -m "feat: assembler uses PlannerContext + per-agent LLM routing with fallback"
```

---

## Task 5: 受控请求生成与冻结评测集

**Files:**
- Create: `backend/ml/planner/__init__.py`（空）、`backend/ml/planner/requestgen.py`、`backend/ml/planner/build_eval_set.py`
- Test: `backend/tests/test_requestgen.py`

**Interfaces:**
- Consumes: `TripRequest`（Task 1）；三个子图（现有）；`build_planner_context`（Task 2）
- Produces:
  - `iter_controlled_requests(count:int, difficulty:str="standard", seed:int=0) -> list[TripRequest]`（确定性）
  - `eval_signature(req: TripRequest) -> str`（防泄漏签名）
  - 评测集 record 格式：`{"record_id": str, "difficulty": str, "request": dict, "context": dict}`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_requestgen.py
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "requestgen", Path(__file__).resolve().parent.parent / "ml" / "planner" / "requestgen.py")
requestgen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(requestgen)


def test_deterministic_with_same_seed():
    a = requestgen.iter_controlled_requests(10, "standard", seed=7)
    b = requestgen.iter_controlled_requests(10, "standard", seed=7)
    assert [r.model_dump(exclude={"user_id"}) for r in a] == \
           [r.model_dump(exclude={"user_id"}) for r in b]


def test_standard_profile():
    for r in requestgen.iter_controlled_requests(20, "standard", seed=1):
        assert 2 <= r.travel_days <= 4
        assert r.party.total <= 2


def test_hard_profile():
    for r in requestgen.iter_controlled_requests(20, "hard", seed=1):
        assert 4 <= r.travel_days <= 6
        assert r.party.total >= 3
        assert r.budget_constraint is not None
        assert r.budget_constraint.strictness == "hard"


def test_signature_stable_and_distinct():
    reqs = requestgen.iter_controlled_requests(30, "standard", seed=2)
    sigs = [requestgen.eval_signature(r) for r in reqs]
    assert sigs[0] == requestgen.eval_signature(reqs[0])
    assert len(set(sigs)) > 25  # 基本不重复
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_requestgen.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 创建 requestgen.py（先 `touch backend/ml/__init__.py backend/ml/planner/__init__.py`）**

```python
# backend/ml/planner/requestgen.py
"""受控请求分布生成器：评测集与训练数据共用，靠 seed 区分与复现。"""
import hashlib
import random
from datetime import date, timedelta
from app.models.schemas import TripRequest, PartyInfo, BudgetConstraint

CITIES = ["北京", "上海", "杭州", "成都", "西安", "广州", "南京", "重庆", "苏州", "厦门"]
PREFS = [["历史文化"], ["美食"], ["自然风光"], ["美食", "城市地标"],
         ["博物馆", "历史文化"], ["亲子"], ["购物", "美食"]]
TRANSPORT = ["公共交通", "打车", "自驾"]
ACCOM = ["经济型酒店", "舒适型酒店", "高档型酒店"]
HARD_FREE_TEXT = ["不吃辣，行程别太赶", "有老人同行，少爬山", "带孩子，需要亲子友好",
                  "素食为主", "不想去人太多的地方"]
_BASE_DATE = date(2026, 8, 1)


def iter_controlled_requests(count: int, difficulty: str = "standard", seed: int = 0) -> list[TripRequest]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        city = CITIES[(i + rng.randrange(len(CITIES))) % len(CITIES)]
        start = _BASE_DATE + timedelta(days=rng.randrange(60))
        if difficulty == "hard":
            days = rng.randint(4, 6)
            party = PartyInfo(adults=2, children=rng.randint(1, 2), elders=rng.randint(0, 1))
            budget = BudgetConstraint(
                amount=days * party.total * rng.choice([300, 400, 500]),
                budget_level="limited", strictness="hard")
            free_text = rng.choice(HARD_FREE_TEXT)
        else:
            days = rng.randint(2, 4)
            party = PartyInfo(adults=rng.randint(1, 2))
            budget = rng.choice([
                None,
                BudgetConstraint(amount=days * party.total * rng.choice([600, 800, 1000]),
                                 budget_level="comfortable", strictness="soft"),
            ])
            free_text = ""
        end = start + timedelta(days=days - 1)
        out.append(TripRequest(
            city=city,
            start_date=start.isoformat(), end_date=end.isoformat(), travel_days=days,
            transportation=rng.choice(TRANSPORT), accommodation=rng.choice(ACCOM),
            preferences=rng.choice(PREFS), free_text_input=free_text,
            party=party, budget_constraint=budget,
        ))
    return out


def eval_signature(req: TripRequest) -> str:
    """请求语义签名，用于训练数据与冻结评测集的防泄漏过滤。"""
    b = req.budget_constraint
    key = "|".join([
        req.city, req.start_date, str(req.travel_days),
        str(req.party.adults), str(req.party.children), str(req.party.elders),
        str(b.amount) if b else "none", b.strictness if b else "none",
        ",".join(sorted(req.preferences)),
    ])
    return hashlib.sha1(key.encode()).hexdigest()[:16]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_requestgen.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: 创建 build_eval_set.py**

```python
# backend/ml/planner/build_eval_set.py
"""构建冻结评测集：真跑三个子图拍候选快照。

运行（在 backend/ 下）：
  python -m ml.planner.build_eval_set --count 50 --difficulty standard --seed 1000 --output ml/planner/eval/records.jsonl
  python -m ml.planner.build_eval_set --count 50 --difficulty hard --seed 2000 --output ml/planner/eval_hard/records.jsonl
支持 --resume：跳过已有 record_id。
"""
import argparse
import asyncio
import json
from pathlib import Path

from app.agents.subgraphs.weather import weather_subgraph
from app.agents.subgraphs.hotel import hotel_subgraph
from app.agents.subgraphs.poi import poi_subgraph
from app.agents.supervisor import _date_range
from app.planner.context import build_planner_context
from ml.planner.requestgen import iter_controlled_requests


async def snapshot_context(req) -> dict:
    weather, hotel, poi = await asyncio.gather(
        weather_subgraph.ainvoke({"city": req.city,
                                  "travel_dates": _date_range(req.start_date, req.travel_days),
                                  "raw_result": "", "weather_result": []}),
        hotel_subgraph.ainvoke({"city": req.city, "accommodation_pref": req.accommodation,
                                "budget_level": "mid", "raw_result": "", "hotel_result": []}),
        poi_subgraph.ainvoke({"city": req.city, "preferences": req.preferences,
                              "travel_days": req.travel_days, "raw_result": "", "poi_result": []}),
    )
    return build_planner_context(
        req, weather["weather_result"], hotel["hotel_result"], poi["poi_result"])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--difficulty", choices=["standard", "hard"], default="standard")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["record_id"] for line in f if line.strip()}

    requests = iter_controlled_requests(args.count, args.difficulty, seed=args.seed)
    with open(out, "a", encoding="utf-8") as f:
        for i, req in enumerate(requests):
            record_id = f"{args.difficulty}_{args.seed}_{i:04d}"
            if record_id in done:
                continue
            try:
                context = await snapshot_context(req)
            except Exception as e:
                print(f"❌ {record_id}: {e}")
                continue
            counts = context["tool_snapshot"]["candidate_counts"]
            if counts["hotels"] == 0 or counts["attractions"] == 0:
                print(f"⚠️ {record_id} 候选不足 {counts}，跳过")
                continue
            f.write(json.dumps({
                "record_id": record_id, "difficulty": args.difficulty,
                "request": req.model_dump(), "context": context,
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"✅ {record_id} ({i + 1}/{len(requests)}) candidates={counts}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: 真跑构建两套评测集（需要 AMAP + DeepSeek key；约 300 次 LLM 解析调用）**

Run（在 backend/ 下）:
```bash
python3 -m ml.planner.build_eval_set --count 50 --difficulty standard --seed 1000 --output ml/planner/eval/records.jsonl --resume
python3 -m ml.planner.build_eval_set --count 50 --difficulty hard --seed 2000 --output ml/planner/eval_hard/records.jsonl --resume
```
Expected: 两个文件各 ~50 行（候选不足的少量跳过可接受，≥45 即可）。抽查 3 条：context 各 section 齐全、hotel/attraction 候选非空。

- [ ] **Step 7: Commit（评测集入 git，从此冻结）**

```bash
git add backend/ml/__init__.py backend/ml/planner/__init__.py backend/ml/planner/requestgen.py \
        backend/ml/planner/build_eval_set.py backend/tests/test_requestgen.py \
        backend/ml/planner/eval/records.jsonl backend/ml/planner/eval_hard/records.jsonl
git commit -m "feat: controlled request generator and frozen standard/hard eval sets"
```

---

## Task 6: 规则评测 rule_eval 与 DeepSeek 基线

**Files:**
- Create: `backend/ml/planner/rule_eval.py`
- Test: `backend/tests/test_rule_eval.py`

**Interfaces:**
- Consumes: `validate_trip_plan / recompute_budget`（Task 3）、`build_planner_messages`（Task 2）、评测集 record（Task 5）
- Produces:
  - `evaluate_output(record: dict, output_text: str) -> dict`（键：`json_ok, schema_ok, violations, hard_pass, meal_repeat_count, budget_ok, soft_pass, recomputed_total`）
  - CLI：`python -m ml.planner.rule_eval --records <path> --base-url <url> --model <name> --api-key-env <ENV> --output-dir <dir>`，产出 `generations.jsonl / report.json / report.md`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_rule_eval.py
import json
import importlib.util
from pathlib import Path
from app.models.schemas import TripRequest, WeatherInfo, Hotel, Attraction, Location
from app.planner.context import build_planner_context

_spec = importlib.util.spec_from_file_location(
    "rule_eval", Path(__file__).resolve().parent.parent / "ml" / "planner" / "rule_eval.py")
rule_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rule_eval)


def _record():
    req = TripRequest(
        city="杭州", start_date="2026-08-01", end_date="2026-08-02", travel_days=2,
        transportation="打车", accommodation="经济型酒店", party={"adults": 2},
        budget_constraint={"amount": 3000, "strictness": "hard"},
    )
    ctx = build_planner_context(
        req,
        [WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云", day_temp=30, night_temp=22),
         WeatherInfo(date="2026-08-02", day_weather="多云", night_weather="多云", day_temp=31, night_temp=23)],
        [Hotel(name="如家杭州店", address="上城区", estimated_cost=250, type="经济型")],
        [Attraction(name="西湖", address="西湖区", location=Location(longitude=120.1, latitude=30.2),
                    visit_duration=120, description="湖", ticket_price=0),
         Attraction(name="灵隐寺", address="西湖区", location=Location(longitude=120.10, latitude=30.24),
                    visit_duration=120, description="寺", ticket_price=30)],
    )
    return {"record_id": "t_0001", "difficulty": "standard",
            "request": req.model_dump(), "context": ctx}


def _good_output():
    def day(i, d, hotel, attraction, ticket, m):
        return {
            "date": d, "day_index": i, "description": f"d{i+1}", "transportation": "打车",
            "accommodation": "经济型酒店", "hotel": hotel,
            "attractions": [{"name": attraction, "address": "西湖区",
                             "location": {"longitude": 120.1, "latitude": 30.2},
                             "visit_duration": 120, "description": "x", "ticket_price": ticket}],
            "meals": [
                {"type": "breakfast", "name": f"{m}豆浆店", "estimated_cost": 15},
                {"type": "lunch", "name": f"{m}面馆", "estimated_cost": 40},
                {"type": "dinner", "name": f"{m}杭帮菜", "estimated_cost": 80},
            ],
        }
    hotel = {"name": "如家杭州店", "address": "上城区", "distance": "", "type": "经济型",
             "estimated_cost": 250}
    return json.dumps({
        "city": "杭州", "start_date": "2026-08-01", "end_date": "2026-08-02",
        "days": [day(0, "2026-08-01", hotel, "西湖", 0, "知味观"),
                 day(1, "2026-08-02", None, "灵隐寺", 30, "外婆家")],
        "weather_info": [
            {"date": "2026-08-01", "day_weather": "晴", "night_weather": "多云", "day_temp": 30, "night_temp": 22},
            {"date": "2026-08-02", "day_weather": "多云", "night_weather": "多云", "day_temp": 31, "night_temp": 23}],
        "overall_suggestions": "ok",
        "budget": {"total_attractions": 60, "total_hotels": 250, "total_meals": 540,
                   "total_transportation": 200, "total": 1050},
    }, ensure_ascii=False)


def test_good_output_hard_and_soft_pass():
    m = rule_eval.evaluate_output(_record(), _good_output())
    assert m["json_ok"] and m["schema_ok"]
    assert m["hard_pass"] is True
    assert m["meal_repeat_count"] == 0
    assert m["budget_ok"] is True
    assert m["soft_pass"] is True


def test_broken_json_fails_hard():
    m = rule_eval.evaluate_output(_record(), "{这不是json")
    assert m["json_ok"] is False and m["hard_pass"] is False


def test_repeated_dinner_fails_soft_only():
    out = json.loads(_good_output())
    out["days"][1]["meals"][2]["name"] = out["days"][0]["meals"][2]["name"]
    m = rule_eval.evaluate_output(_record(), json.dumps(out, ensure_ascii=False))
    assert m["hard_pass"] is True
    assert m["meal_repeat_count"] == 1
    assert m["soft_pass"] is False


def test_over_budget_fails_soft():
    out = json.loads(_good_output())
    out["days"][0]["hotel"]["estimated_cost"] = 5000
    m = rule_eval.evaluate_output(_record(), json.dumps(out, ensure_ascii=False))
    assert m["budget_ok"] is False and m["soft_pass"] is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_rule_eval.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 创建 rule_eval.py**

```python
# backend/ml/planner/rule_eval.py
"""规则评测：对任意 OpenAI-compatible 端点跑冻结评测集，输出 hardpass/softpass。

运行（在 backend/ 下）：
  python -m ml.planner.rule_eval --records ml/planner/eval/records.jsonl \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat --api-key-env DEEPSEEK_API_KEY \
    --output-dir ml/planner/runs_eval/deepseek_standard
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from app.models.schemas import TripPlan
from app.planner.context import build_planner_messages
from app.planner.validation import validate_trip_plan, recompute_budget


def _strip_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def evaluate_output(record: dict, output_text: str) -> dict:
    m = {"record_id": record["record_id"], "json_ok": False, "schema_ok": False,
         "violations": [], "hard_pass": False, "meal_repeat_count": 0,
         "budget_ok": True, "recomputed_total": None, "soft_pass": False}
    try:
        data = json.loads(_strip_fences(output_text))
        m["json_ok"] = True
        plan = TripPlan(**data)
        m["schema_ok"] = True
    except Exception as e:
        m["violations"] = [f"parse: {e}"]
        return m

    ctx = record["context"]
    m["violations"] = validate_trip_plan(plan, ctx)
    m["hard_pass"] = not m["violations"]

    names = [meal.name for d in plan.days for meal in d.meals if meal.type in ("lunch", "dinner")]
    m["meal_repeat_count"] = len(names) - len(set(names))

    budget = recompute_budget(plan, ctx["party"]["total"])
    m["recomputed_total"] = budget.total
    bc = ctx["budget_constraint"]
    if bc["amount"]:
        if bc["strictness"] == "hard":
            m["budget_ok"] = budget.total <= bc["amount"] and budget.total >= 0.4 * bc["amount"]
        else:
            m["budget_ok"] = budget.total <= 1.2 * bc["amount"]
    m["soft_pass"] = m["hard_pass"] and m["meal_repeat_count"] == 0 and m["budget_ok"]
    return m


def aggregate(metrics: list[dict]) -> dict:
    n = len(metrics)
    rate = lambda k: round(sum(1 for x in metrics if x[k]) / n * 100, 1) if n else 0.0
    return {"count": n, "json_ok": rate("json_ok"), "schema_ok": rate("schema_ok"),
            "hard_pass": rate("hard_pass"), "soft_pass": rate("soft_pass"),
            "budget_ok": rate("budget_ok"),
            "meal_repeat_avg": round(sum(x["meal_repeat_count"] for x in metrics) / n, 2) if n else 0}


async def _generate(llm: ChatOpenAI, record: dict, sem: asyncio.Semaphore) -> tuple[dict, str]:
    async with sem:
        resp = await llm.ainvoke(build_planner_messages(record["context"]))
        return record, resp.content


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.records, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    llm = ChatOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"),
                     model=args.model, temperature=args.temperature,
                     max_tokens=args.max_tokens, timeout=300)
    sem = asyncio.Semaphore(args.workers)
    results = await asyncio.gather(
        *[_generate(llm, r, sem) for r in records], return_exceptions=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            if isinstance(item, Exception):
                print(f"❌ 生成失败: {item}")
                continue
            record, text = item
            m = evaluate_output(record, text)
            metrics.append(m)
            f.write(json.dumps({"record_id": record["record_id"], "output": text,
                                "metrics": m}, ensure_ascii=False) + "\n")

    report = {"model": args.model, "records": args.records, "summary": aggregate(metrics)}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    s = report["summary"]
    md = (f"# Rule Eval: {args.model}\n\n| 指标 | 值 |\n| --- | ---: |\n"
          f"| 样本数 | {s['count']} |\n| json_ok | {s['json_ok']}% |\n"
          f"| schema_ok | {s['schema_ok']}% |\n| **hard_pass** | **{s['hard_pass']}%** |\n"
          f"| **soft_pass** | **{s['soft_pass']}%** |\n| budget_ok | {s['budget_ok']}% |\n"
          f"| 午晚餐平均重复 | {s['meal_repeat_avg']} |\n")
    (out_dir / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_rule_eval.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: 真跑 DeepSeek 基线（standard + hard）**

Run（在 backend/ 下）:
```bash
python3 -m ml.planner.rule_eval --records ml/planner/eval/records.jsonl \
  --base-url https://api.deepseek.com/v1 --model deepseek-chat \
  --output-dir ml/planner/runs_eval/deepseek_standard
python3 -m ml.planner.rule_eval --records ml/planner/eval_hard/records.jsonl \
  --base-url https://api.deepseek.com/v1 --model deepseek-chat \
  --output-dir ml/planner/runs_eval/deepseek_hard
```
Expected: 两份 report.md。把两份数字合并抄进 `backend/ml/planner/reports/260703_baseline_deepseek.md`（这是后续所有对比的基线）。

- [ ] **Step 6: Commit**

```bash
git add backend/ml/planner/rule_eval.py backend/tests/test_rule_eval.py \
        backend/ml/planner/reports/260703_baseline_deepseek.md
git commit -m "feat: rule eval (hardpass/softpass) with DeepSeek baseline report"
```

---

## Task 7: teacher SFT 数据生成（run 目录 + manifest + usage + 防泄漏）

**Files:**
- Create: `backend/ml/planner/data_gen.py`
- Test: `backend/tests/test_data_gen_planner.py`

**Interfaces:**
- Consumes: `iter_controlled_requests / eval_signature`（Task 5）、`snapshot_context`（Task 5 的 build_eval_set）、`evaluate_output`（Task 6）、`acall_with_fallback`（现有，teacher=DeepSeek）
- Produces:
  - `load_eval_signatures(paths: list[str]) -> set[str]`
  - run 目录产物：`requests.jsonl / records.jsonl / errors.jsonl / manifest.json`
  - records 行格式与评测集一致 + `"teacher_output": str`（通过硬过滤的 plan JSON 文本）

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_data_gen_planner.py
import json
import importlib.util
from pathlib import Path

_here = Path(__file__).resolve().parent.parent / "ml" / "planner"
_spec = importlib.util.spec_from_file_location("planner_data_gen", _here / "data_gen.py")
data_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_gen)

_rspec = importlib.util.spec_from_file_location("requestgen", _here / "requestgen.py")
requestgen = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(requestgen)


def test_load_eval_signatures(tmp_path):
    reqs = requestgen.iter_controlled_requests(3, "standard", seed=5)
    p = tmp_path / "records.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, r in enumerate(reqs):
            f.write(json.dumps({"record_id": f"s_{i}", "difficulty": "standard",
                                "request": r.model_dump(), "context": {}},
                               ensure_ascii=False) + "\n")
    sigs = data_gen.load_eval_signatures([str(p)])
    assert len(sigs) == 3
    assert requestgen.eval_signature(reqs[0]) in sigs


def test_extract_usage_from_metadata():
    class R:
        response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    u = data_gen.extract_usage(R())
    assert u == {"prompt_tokens": 100, "completion_tokens": 50}

    class Empty:
        response_metadata = {}
    assert data_gen.extract_usage(Empty()) == {"prompt_tokens": 0, "completion_tokens": 0}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_data_gen_planner.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 创建 data_gen.py**

```python
# backend/ml/planner/data_gen.py
"""teacher（DeepSeek）生成 SFT 数据：子图快照 → teacher 生成 → 规则硬过滤。

运行（在 backend/ 下）：
  python -m ml.planner.data_gen --count 20 --seed 9000 --run-slug 260703_smoke20
节奏：smoke 20 → 人工审计 → 100 → 审计 → 1000。每个 run 独立目录 + manifest + usage。
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.agents.llm_router import acall_with_fallback
from app.planner.context import build_planner_messages
from ml.planner.requestgen import iter_controlled_requests, eval_signature
from ml.planner.build_eval_set import snapshot_context
from ml.planner.rule_eval import evaluate_output, _strip_fences

EVAL_PATHS = ["ml/planner/eval/records.jsonl", "ml/planner/eval_hard/records.jsonl"]
RUNS_DIR = Path("ml/planner/data/runs")


def load_eval_signatures(paths: list[str]) -> set[str]:
    from app.models.schemas import TripRequest
    sigs = set()
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sigs.add(eval_signature(TripRequest(**json.loads(line)["request"])))
    return sigs


def extract_usage(response) -> dict:
    u = getattr(response, "response_metadata", {}).get("token_usage", {})
    return {"prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="必须与评测集 seed(1000/2000) 不同")
    ap.add_argument("--run-slug", required=True, help="如 260703_smoke20")
    ap.add_argument("--hard-ratio", type=float, default=0.2)
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_sigs = load_eval_signatures(EVAL_PATHS)

    n_hard = int(args.count * args.hard_ratio)
    requests = (iter_controlled_requests(args.count - n_hard, "standard", seed=args.seed)
                + iter_controlled_requests(n_hard, "hard", seed=args.seed + 1))

    done = set()
    records_path = run_dir / "records.jsonl"
    if records_path.exists():
        with open(records_path, encoding="utf-8") as f:
            done = {json.loads(line)["record_id"] for line in f if line.strip()}

    stats = {"requested": len(requests), "eval_overlap_skipped": 0, "context_failed": 0,
             "hard_pass": 0, "hard_fail": 0,
             "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    with open(run_dir / "requests.jsonl", "w", encoding="utf-8") as freq, \
         open(records_path, "a", encoding="utf-8") as frec, \
         open(run_dir / "errors.jsonl", "a", encoding="utf-8") as ferr:
        for i, req in enumerate(requests):
            difficulty = "hard" if i >= len(requests) - n_hard else "standard"
            record_id = f"sft_{args.run_slug}_{i:04d}"
            freq.write(json.dumps({"record_id": record_id, "request": req.model_dump()},
                                  ensure_ascii=False) + "\n")
            if record_id in done:
                continue
            if eval_signature(req) in eval_sigs:
                stats["eval_overlap_skipped"] += 1
                continue
            try:
                context = await snapshot_context(req)
            except Exception as e:
                stats["context_failed"] += 1
                ferr.write(json.dumps({"record_id": record_id, "stage": "context",
                                       "error": str(e)}, ensure_ascii=False) + "\n")
                continue

            response = await acall_with_fallback(build_planner_messages(context))
            usage = extract_usage(response)
            stats["usage"]["prompt_tokens"] += usage["prompt_tokens"]
            stats["usage"]["completion_tokens"] += usage["completion_tokens"]

            record = {"record_id": record_id, "difficulty": difficulty,
                      "request": req.model_dump(), "context": context}
            metrics = evaluate_output(record, response.content)
            if metrics["hard_pass"]:
                stats["hard_pass"] += 1
                record["teacher_output"] = _strip_fences(response.content)
                frec.write(json.dumps(record, ensure_ascii=False) + "\n")
                frec.flush()
            else:
                stats["hard_fail"] += 1
                ferr.write(json.dumps({"record_id": record_id, "stage": "audit",
                                       "violations": metrics["violations"][:10],
                                       "output": response.content},
                                      ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(requests)}] {record_id} hard_pass={metrics['hard_pass']}")

    manifest = {"run_slug": args.run_slug, "created_at": datetime.now().isoformat(),
                "seed": args.seed, "hard_ratio": args.hard_ratio, "stats": stats}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_data_gen_planner.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: 真跑 smoke 20 并人工审计**

Run: `cd backend && python3 -m ml.planner.data_gen --count 20 --seed 9000 --run-slug 260703_smoke20`
Expected: manifest 打印通过率与 usage。审计：抽 5 条 records 逐条核对（酒店连续、门票×人数、餐饮非占位、hard 预算不超）；通过率 < 60% 时先改 `PLANNER_SYSTEM_PROMPT` 或过滤规则，再进入下一步。

- [ ] **Step 6: 真跑 100 条 → 审计 → 1000 条**

```bash
python3 -m ml.planner.data_gen --count 100 --seed 9100 --run-slug 260704_batch100
# 审计通过后：
python3 -m ml.planner.data_gen --count 1000 --seed 9200 --run-slug 260705_batch1000
```
Expected: 最终合计 hard_pass 记录 ≥ 800 条。每轮审计结论（通过率、典型失败、usage 成本）追记到 `backend/ml/planner/reports/260703_sft_data_runs.md`。

- [ ] **Step 7: Commit（数据在 gitignore 内，只提交脚本、测试与报告）**

```bash
git add backend/ml/planner/data_gen.py backend/tests/test_data_gen_planner.py \
        backend/ml/planner/reports/260703_sft_data_runs.md
git commit -m "feat: teacher SFT data generation with run manifests, usage and leakage guard"
```

---

## Task 8: LLaMA-Factory 导出

**Files:**
- Create: `backend/ml/planner/export_llamafactory.py`
- Create: `backend/ml/planner/llamafactory/dataset_info.json`
- Test: `backend/tests/test_export_llamafactory.py`

**Interfaces:**
- Consumes: data run 的 `records.jsonl`（Task 7，含 `teacher_output`）、`PLANNER_SYSTEM_PROMPT / build_planner_messages`（Task 2）
- Produces:
  - `to_sharegpt(record: dict) -> dict`（`{"conversations": [{"from": "human", ...}, {"from": "gpt", ...}], "system": ...}`）
  - `split_rows(rows: list, val_ratio: float = 0.05, seed: int = 42) -> tuple[list, list]`
  - `llamafactory/generated/train.json` 与 `val.json`（gitignore 内）

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_export_llamafactory.py
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "export_lf", Path(__file__).resolve().parent.parent / "ml" / "planner" / "export_llamafactory.py")
export_lf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_lf)


def test_to_sharegpt_structure():
    record = {"record_id": "x", "context": {"request": {"city": "北京"}},
              "teacher_output": '{"city":"北京"}'}
    row = export_lf.to_sharegpt(record)
    assert set(row.keys()) == {"conversations", "system"}
    assert row["conversations"][0]["from"] == "human"
    assert "北京" in row["conversations"][0]["value"]
    assert row["conversations"][1] == {"from": "gpt", "value": '{"city":"北京"}'}


def test_split_deterministic_and_disjoint():
    rows = [{"id": i} for i in range(100)]
    t1, v1 = export_lf.split_rows(rows, val_ratio=0.05, seed=42)
    t2, v2 = export_lf.split_rows(rows, val_ratio=0.05, seed=42)
    assert t1 == t2 and v1 == v2
    assert len(v1) == 5
    assert not {id(x) for x in t1} & {id(x) for x in v1}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_export_llamafactory.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 创建 export_llamafactory.py 与 dataset_info.json**

```python
# backend/ml/planner/export_llamafactory.py
"""把审计通过的 teacher 数据导出为 LLaMA-Factory sharegpt 格式。

运行（在 backend/ 下）：
  python -m ml.planner.export_llamafactory \
    --runs 260703_smoke20 260704_batch100 260705_batch1000
产出：ml/planner/llamafactory/generated/train.json / val.json
"""
import argparse
import json
import random
from pathlib import Path

from app.planner.context import PLANNER_SYSTEM_PROMPT, build_planner_messages

RUNS_DIR = Path("ml/planner/data/runs")
OUT_DIR = Path("ml/planner/llamafactory/generated")


def to_sharegpt(record: dict) -> dict:
    human = build_planner_messages(record["context"])[1].content
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": record["teacher_output"]},
        ],
        "system": PLANNER_SYSTEM_PROMPT,
    }


def split_rows(rows: list, val_ratio: float = 0.05, seed: int = 42):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(rows) * val_ratio))
    val_idx = set(idx[:n_val])
    train = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val = [rows[i] for i in range(len(rows)) if i in val_idx]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    args = ap.parse_args()

    rows = []
    for slug in args.runs:
        path = RUNS_DIR / slug / "records.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(to_sharegpt(json.loads(line)))
        print(f"{slug}: 累计 {len(rows)} 条")

    train, val = split_rows(rows, args.val_ratio)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=1))
    (OUT_DIR / "val.json").write_text(json.dumps(val, ensure_ascii=False, indent=1))
    print(f"train={len(train)} val={len(val)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
```

```json
// backend/ml/planner/llamafactory/dataset_info.json
{
  "trip_planner_sft": {
    "file_name": "generated/train.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system"}
  },
  "trip_planner_sft_val": {
    "file_name": "generated/val.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system"}
  }
}
```

- [ ] **Step 4: 运行测试确认通过 + 真跑导出**

Run: `cd backend && python3 -m pytest tests/test_export_llamafactory.py -q`
Expected: PASS (2 passed)

Run: `cd backend && python3 -m ml.planner.export_llamafactory --runs 260703_smoke20 260704_batch100 260705_batch1000`
Expected: 打印 train/val 条数（train ≥ 760）。

- [ ] **Step 5: Commit**

```bash
git add backend/ml/planner/export_llamafactory.py backend/ml/planner/llamafactory/dataset_info.json \
        backend/tests/test_export_llamafactory.py
git commit -m "feat: export audited SFT data to LLaMA-Factory sharegpt format"
```

---

## Task 9: LoRA SFT 训练（GPU 环境）

**Files:**
- Create: `backend/ml/planner/configs/qwen25_7b_lora_sft.yaml`

**Interfaces:**
- Consumes: `dataset_info.json` 与 generated train/val（Task 8）
- Produces: LoRA adapter `backend/ml/planner/outputs/qwen25_7b_sft_v1/`（gitignore 内）

- [ ] **Step 1: 创建训练配置**

```yaml
# backend/ml/planner/configs/qwen25_7b_lora_sft.yaml
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

### dataset
dataset: trip_planner_sft
eval_dataset: trip_planner_sft_val
dataset_dir: ml/planner/llamafactory
template: qwen
cutoff_len: 8192
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: ml/planner/outputs/qwen25_7b_sft_v1
logging_steps: 5
save_steps: 100
save_total_limit: 4
plot_loss: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
```

- [ ] **Step 2: 在 GPU 环境安装 LLaMA-Factory 并启动训练**

前提：1×24GB GPU 可跑（8k ctx + LoRA + bf16 + gradient checkpointing，贴边）；1×40GB 更稳。LLaMA-Factory 装在独立环境，**不进 backend requirements**。

```bash
pip install llamafactory  # 或 git clone LLaMA-Factory 后 pip install -e .
cd backend
llamafactory-cli train ml/planner/configs/qwen25_7b_lora_sft.yaml
```
Expected: 训练正常推进（loss 下降、eval_loss 每 50 step 打印且趋稳、不 OOM），`ml/planner/outputs/qwen25_7b_sft_v1/` 下有 checkpoint。eval_loss 只看训练是否坏了，**选点以 Task 10 的规则评测为准**。

- [ ] **Step 3: 合并 LoRA 供 vLLM 使用**

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path ml/planner/outputs/qwen25_7b_sft_v1 \
  --template qwen \
  --export_dir ml/planner/outputs/qwen25_7b_sft_v1_merged
```
Expected: merged 目录含完整权重与 tokenizer。

- [ ] **Step 4: Commit（只提交 config）**

```bash
git add backend/ml/planner/configs/qwen25_7b_lora_sft.yaml
git commit -m "feat: add LLaMA-Factory LoRA SFT config for planner model"
```

---

## Task 10: vLLM 服务、三方对比评测与达标接入

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/agents_config.yaml`（仅达标后）
- Modify: `README.md`
- Create: `backend/ml/planner/reports/260707_sft_vs_baselines.md`（评测后）

**Interfaces:**
- Consumes: merged 模型（Task 9）、`rule_eval` CLI（Task 6）、`acall_agent_with_fallback`（Task 4，代码已就绪，本任务只改配置）

- [ ] **Step 1: config.py 增加 local provider**

在 `Settings` 的 OpenAI 配置段之后新增：

```python
    # 本地微调模型（vLLM OpenAI-compatible）
    local_api_key: str = "EMPTY"
    local_base_url: str = "http://127.0.0.1:8001/v1"
    local_model: str = "trip-planner-sft"
```

Run: `cd backend && python3 -m pytest tests/ -q` → 全绿后提交：

```bash
git add backend/app/config.py
git commit -m "feat: add local vLLM provider settings for finetuned planner"
```

- [ ] **Step 2: 启动 vLLM 服务（GPU 环境）**

```bash
vllm serve backend/ml/planner/outputs/qwen25_7b_sft_v1_merged \
  --served-model-name trip-planner-sft --port 8001 --max-model-len 16384
```
Expected: `curl http://127.0.0.1:8001/v1/models` 返回 `trip-planner-sft`。

- [ ] **Step 3: 三方对比评测（standard + hard 各三跑）**

```bash
cd backend
# 未微调基座（vLLM 另起: vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name qwen-base --port 8002）
python3 -m ml.planner.rule_eval --records ml/planner/eval/records.jsonl \
  --base-url http://127.0.0.1:8002/v1 --model qwen-base --api-key-env NONE \
  --output-dir ml/planner/runs_eval/qwen_base_standard
python3 -m ml.planner.rule_eval --records ml/planner/eval_hard/records.jsonl \
  --base-url http://127.0.0.1:8002/v1 --model qwen-base --api-key-env NONE \
  --output-dir ml/planner/runs_eval/qwen_base_hard
# SFT 模型
python3 -m ml.planner.rule_eval --records ml/planner/eval/records.jsonl \
  --base-url http://127.0.0.1:8001/v1 --model trip-planner-sft --api-key-env NONE \
  --output-dir ml/planner/runs_eval/sft_v1_standard
python3 -m ml.planner.rule_eval --records ml/planner/eval_hard/records.jsonl \
  --base-url http://127.0.0.1:8001/v1 --model trip-planner-sft --api-key-env NONE \
  --output-dir ml/planner/runs_eval/sft_v1_hard
```

把 DeepSeek（Task 6）/ qwen-base / sft_v1 六份 report 汇总成对比表，写入 `backend/ml/planner/reports/260707_sft_vs_baselines.md`。

**验收（Global Constraints 的成功标准）：** sft_v1 hardpass standard ≥ 85% 且 hard ≥ 70%；比 qwen-base 高 ≥ 20pp；不低于 DeepSeek −5pp。
**未达标时**：不切配置；按失败画像回 Task 7 追加数据（针对 top violations 类型），重训 v2 —— 评测集不动。

- [ ] **Step 4: 达标后切换 assembler 到本地模型**

`backend/agents_config.yaml` 中 `assembler` 段改为：

```yaml
  assembler:
    provider: local
    model: trip-planner-sft
    temperature: 0.2
```

冒烟验证：启动 backend，POST `/api/trip/plan` 一条真实请求，确认返回计划且日志无降级；随后停掉 vLLM 再请求一次，确认自动降级 DeepSeek 且服务不中断。

- [ ] **Step 5: README 增加「行程生成模型（微调）」小节**

在 README「🔧 核心实现」末尾追加：

```markdown
### 行程生成模型（微调）

行程 JSON 由 LoRA 微调的 Qwen2.5-7B-Instruct（vLLM 本地服务）生成，DeepSeek 自动兜底。
复现流程（backend/ 下）：

​```bash
python -m ml.planner.build_eval_set --count 50 --difficulty standard --seed 1000 --output ml/planner/eval/records.jsonl   # 冻结评测集（已入库，勿重采样）
python -m ml.planner.data_gen --count 20 --seed 9000 --run-slug <slug>       # teacher 数据（smoke → 100 → 1000，逐轮审计）
python -m ml.planner.export_llamafactory --runs <slug...>                    # 导出 LLaMA-Factory 格式
llamafactory-cli train ml/planner/configs/qwen25_7b_lora_sft.yaml            # LoRA SFT
python -m ml.planner.rule_eval --records ml/planner/eval/records.jsonl ...   # 规则评测（hardpass/softpass）
​```

评测集冻结在 `ml/planner/eval*/records.jsonl`；数据 run 带 manifest 与 token usage；
训练数据经 `eval_signature` 过滤，与评测集零重叠。
```

- [ ] **Step 6: 最终回归与 Commit**

Run: `cd backend && python3 -m pytest tests/ -q` → 全绿。

```bash
git add backend/agents_config.yaml README.md backend/ml/planner/reports/260707_sft_vs_baselines.md
git commit -m "feat: switch assembler to finetuned local planner model with DeepSeek fallback"
```

---

## Self-Review 记录

- **Spec 覆盖**：协议改造（T1-T4）、冻结评测集与规则评测（T5-T6）、teacher 数据 + 审计 + manifest/usage/防泄漏（T7）、导出与训练口径（T8-T9）、vLLM 接入 + 三方对比 + 达标门槛 + 兜底验证（T10）；spec 的错误处理（降级、校验只告警、预算重算）落在 T3/T4/T10；非目标（DPO/rerank/前端表单）未混入。
- **Placeholder 扫描**：所有代码任务给出完整实现；训练/评测的"真跑"步骤给出确切命令与量化验收标准（通过率、hardpass 阈值），属实验步骤而非代码占位。
- **类型一致性**：`build_planner_context` 七个 section 键名在 context/validation/rule_eval/data_gen/export 中一致；record 格式（record_id/difficulty/request/context [+teacher_output]）在 T5/T6/T7/T8 一致；`evaluate_output` 返回键在 T6 定义、T7 消费；`acall_agent_with_fallback` 在 T4 定义、T10 仅配置切换；`eval_signature` 在 T5 定义、T7 消费。
- **与意图微调计划的交集**：两计划都改 `backend/.gitignore` 与 README，均为追加不冲突；本计划不动 `ml/intent/`，不引入 torch 到 backend requirements。
