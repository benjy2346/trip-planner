# Grounded Planner 集成设计（Grounded MVP）

> 状态：设计稿，待用户 review。参照项目 `helloagents-trip-planner`（同 `~/Desktop/UCLA/` 下），**每一步以他的对应文件为准**。

## 1. 目标

让我们 `trip-planner` app 的行程生成用上一个 **grounded 的 LoRA 微调 Planner 模型**：后端把请求 + 真实工具候选编译成结构化 `PlannerContext`，模型只在候选与约束内生成 `TripPlan JSON`，DeepSeek 常驻兜底。

对齐 helloagents 的核心思想：**Planner = 业务编译器，尽量少猜**；事实由后端做实，模型只做选择与组装。

## 2. 问题（为什么重做）

当前实现的 grounding 是空壳（已实测）：
- 景点坐标 **93% 为 `0,0`**、酒店 `location` 56% 为 null —— 根因：子图用 LLM 解析高德返回、缺字段就填 0。
- 门票 **98% 为 0**，预算里「景点」项恒等于 0。
- **三餐无候选**，模型凭记忆编店名（会过时、会幻觉、陌生城市露馅）。
- 天气几乎全空（请求日期全在预报窗外），评测测不到。
- 评测集只覆盖热门城市 + 远期日期，**天然藏住上述问题**。

结论：模型被训成「按格式输出空壳」。必须先把 context 做实。

## 3. 范围（Grounded MVP）

**做**：真坐标 + 餐饮候选（grounded）+ 真价格 hint + `preference_profile`（把 free_text 解析成结构化饮食/负向/节奏约束）+ mixed 日期（含天气）。

**不做（延后）**：`high_end_candidates`（预算档升级）、`rerank`（best-of-n，第 4 阶段）、DPO（第 3 阶段）、`output.py` 富化渲染。

**成功标准**：
- grounded 评测集上，SFT 模型 hardpass **standard ≥ 85% / hard ≥ 70%**，比基座 +20pp，不低于 DeepSeek −5pp。
- 新增 **餐饮 grounding 指标**（三餐店名 ∈ food_candidates 的比例）与 **坐标非空率**，SFT 显著优于基座。
- 线上：assembler 走本地模型，停 vLLM 能自动降级 DeepSeek。

## 4. 架构

核心是一个**独立的 `PlannerContextBuilder`**（镜像 helloagents `backend/app/planner/context.py` 的 `PlannerContextBuilder`），同步直接调高德取结构化数据，拼出 grounded `PlannerContext`。

**关键不变式：数据生成与线上推理共用同一个 Builder** → train/serve 天然一致。

多 agent 保留但退化为薄壳：`supervisor` 的取数改为**调用同一个 Builder**（不再用 LLM 解析子图），`assembler_node` 保持不变（已在 T4 接 `acall_agent_with_fallback` + 本地/DeepSeek 路由）。编排风格不影响模型（模型只见 `PlannerContext→TripPlan`）。

```
request
  → policy.build_preference_profile()      # free_text → 结构化约束
  → PlannerContextBuilder.collect():
        amap 结构化搜索(景点/酒店/餐饮, 真坐标)
        + pricing hint(门票/餐价/房价)
        + 饮食约束过滤餐饮候选
        → PlannerContext(含 food_candidates + preference_profile + 真坐标/价)
  → [数据生成: DeepSeek teacher] / [线上: SFT模型, DeepSeek兜底]
  → TripPlan JSON
  → validation(景点/酒店/餐饮 grounding + 饮食约束 + 预算重算)
```

## 5. 组件与 helloagents 参照映射

| 我们的动作 | 目标文件 | 参照 helloagents 文件 |
| --- | --- | --- |
| 搬入 | `app/planner/amap.py`（结构化高德客户端） | `backend/app/planner/amap.py` |
| 搬入 | `app/planner/pois.py`（POI+餐饮搜索/关键词组/饮食过滤） | `backend/app/planner/pois.py` |
| 搬入 | `app/planner/policy.py`（`preference_profile` 解析） | `backend/app/planner/policy.py` |
| 合并 | `app/planner/pricing.py`（票价/餐价/房价 hint） | `backend/app/planner/pricing.py` |
| 重写 | `app/planner/context.py` → `PlannerContextBuilder`（+food_candidates +preference_profile +route_policy +餐饮 grounding prompt 规则） | `backend/app/planner/context.py` |
| 扩写 | `app/planner/validation.py`（三餐 grounding + 饮食约束校验） | 参照他的校验/规则 eval |
| 改写 | `app/agents/supervisor.py`（取数改调 Builder） | `backend/app/agents/trip_planner_agent.py` 的 `plan_trip` |
| 弃用 | `app/agents/subgraphs/{poi,hotel,weather}.py`（LLM 解析） | —— |
| 不搬 | high_end_candidates / rerank / output | —— |

**schema**：已基本对齐（`PartyInfo/BudgetConstraint/Attraction(location,ticket_price)/Meal(location)/Hotel/DayPlan/Budget/TripPlan` 我们都有，当初照他抄的）。仅按需补 `preference_profile` 相关字段。

## 6. 数据 + 评测重建

旧的 783 条 SFT 数据、44+39 评测集**全部弃用**（薄格式）。

- **请求分布**：参照他 `training/scripts/planner/data/generate_sft_data.py` 的受控分布 —— `--date-mode mixed`（past/near/far，near 覆盖天气）、city_tier（含 long_tail）、companion_type。
- **数据生成**：新 grounded Builder + DeepSeek teacher → 硬过滤 + 餐厅去重过滤（沿用我们已建的过滤）。节奏 smoke20 → 审计 → 100 → 审计 → 目标量。
- **评测集**：同 Builder 重建 standard/hard，冻结入 git，与训练数据 `eval_signature` 零重叠。
- **评测框架**：复用 `ml/planner/rule_eval.py`，新增 `meal_grounding_rate`、`coord_nonzero_rate` 指标。

## 7. 训练 / 上线

- LoRA SFT：现有 `configs/qwen25_7b_lora_sft.yaml`（Qwen2.5-7B, r32/α64, 8k ctx, bf16, grad ckpt）。GPU = **RTX 5090 D 32GB**，峰值 ~20GB，绰绰有余。
- 合并 LoRA → vLLM 服务（`trip-planner-sft`）。
- `agents_config.yaml` 达标后切 assembler → local，DeepSeek 兜底（T4 代码已就位）。

## 8. 测试（TDD）

每个搬入/重写模块先写失败测试：
- `amap.py`：结构化响应解析出真坐标（mock 高德返回）。
- `pois.py`：候选带非零坐标；餐饮按饮食约束过滤。
- `policy.py`：free_text（"不吃辣/带老人"）→ 正确的 `preference_profile`。
- `pricing.py`：票价/餐价 hint 非零且符合口径。
- `context.py`：`PlannerContext` 含 food_candidates + preference_profile，train/serve 同源。
- `validation.py`：餐饮 grounding、饮食约束、坐标非空校验。

现有测试保持全绿。

## 9. 分步落地（供 writing-plans 展开）

1. 搬 `amap.py` + `pois.py`（结构化取数，真坐标 + 餐饮候选）
2. 搬 `policy.py`（preference_profile）+ 合并 `pricing.py`
3. 重写 `context.py` 为 `PlannerContextBuilder`（含新段 + prompt 规则）
4. 扩 `validation.py`（餐饮 grounding + 饮食约束）
5. supervisor 取数改调 Builder，弃子图；回归测试
6. 参照他的请求分布重建 requestgen + 评测集
7. 重生成 SFT 数据（smoke→审计→量产）+ 导出
8. LoRA 训练（5090 D）+ 三方对比评测 + 达标切配置

## 10. 已知延后（非本 MVP）

high_end_candidates（预算档）、rerank（best-of-n）、DPO（偏好优化，治 softpass）、output 富化渲染。这些是 helloagents 的后续阶段，v1 达标后再按需迭代。
