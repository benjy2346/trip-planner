# Grounded Planner 训练数据生成（Plan 2a）设计

> Plan 2 拆分的前半：**数据生成**（本地、纯 API、无 GPU）。后半 **2b（LoRA 训练 + 三方评测）** 需远程 GPU/SSH，另出 spec。

## 背景与目标

Plan 1（grounded planner 后端）已完成：`PlannerContextBuilder` 产出真坐标 + 餐饮候选 + 真价格 hint + `preference_profile` 的 grounded `PlannerContext`；`validate_grounded_trip_plan` 做输出侧硬校验；supervisor 已接线，E2E 验证过（`ff6dd44`）。

旧的 thin-SFT 数据（783 条）已废弃——它建在 hollow context 上（93% 坐标 0,0、无餐饮候选、天气空）。**2a 的目标：用 grounded 取数 + DeepSeek teacher，生成一批干净的 `(PlannerContext → TripPlan)` 训练样本**，导出为 LLaMA-Factory sharegpt 格式，供 2b 微调 Qwen2.5-7B。

**source of truth**：helloagents `training/scripts/planner/data/generate_sft_data.py`（2775 行）。原则：**参照它、写我们自己的，不逐字照抄**（prompt 文本是唯一例外，Plan 1 已逐字移植）。

## 产出

- `ml/planner/data/runs/<slug>/`：每批一个目录（`records.jsonl` / `requests.jsonl` / `errors.jsonl` / `manifest.json`）。
- `ml/planner/llamafactory/generated/train.json` + `val.json`（sharegpt，grounded，gitignored）。
- 拷入冻结评测集：他的 `eval`(200) / `eval_hard`(300) grounded 记录，committed，冻结不动。

## 范围

**在 2a 内：**
- Grounded 数据生成器（新写，参照他）。
- `DataGenPlannerContextBuilder`：训练/远期日期用 open-meteo 历史天气 override。
- 导出脚本接 grounded 记录 + grounded 校验。
- 拷入他的 200/300 评测集。
- 分阶段跑：smoke20 → 审计 → 100 → 审计 → ~1200 生（目标 ~800-1000 干净）。

**不在 2a 内（留给 2b，需 GPU）：** LoRA 训练配置/训练、vLLM serve、三方评测（用他的 `eval_rule_metrics.py`）、模型切换。

## 架构与组件

### 组件 1：控制变量请求生成器（参照他，弃用旧 `requestgen`）

加权采样城市/日期/同行人/预算档/饮食/偏好，产出 `TripRequest` + `control_spec`（记录控制变量，供 2b 评测切片）。旧 `ml/planner/requestgen.py` 让位（它请求维度太薄）。

- **`eval_signature` 等价物**：泄漏 guard 需要给请求算指纹；参照他的实现提供一个等价函数。

### 组件 2：`DataGenPlannerContextBuilder(PlannerContextBuilder)`（参照他）

继承 Plan 1 的 Builder，只 override `_collect_weather_snapshot`：
- 近期日期（高德预报覆盖内）→ 仍走高德 forecast（Plan 1 现状）。
- 远期日期 → 走 **open-meteo 历史档案 API**（免费、无 key），取往年同期天气作为 grounded 天气。
- 参照他的 `throttle_open_meteo_call` 加限流，避免打爆免费接口。

**为什么必须**：训练请求日期是变化/远期的，高德无预报 → 不 override 则天气段全空，正是审计要修的缺陷。

### 组件 3：teacher 生成循环（复用我们脚手架 + 换 grounded 三段）

复用 `data_gen.py` 的长跑脚手架，换掉接在死路上的三段：

| 环节 | 旧（死路） | 2a（grounded） |
|---|---|---|
| 取数 | `snapshot_context`（旧子图，坐标 0,0） | `DataGenPlannerContextBuilder.collect` |
| 消息 | `build_planner_messages`（旧 prompt） | `build_grounded_planner_messages`（Plan 1） |
| 清洗判据 | `evaluate_output`（旧规则） | `validate_grounded_trip_plan`（Plan 1，无违规=干净） |

**复用不动的脚手架**（与取数路径无关，纯保命机制）：
- **manifest**：每批 `manifest.json` 记 run/seed/stats/usage 台账。
- **resume**：重启读已有 `records.jsonl` 跳过已完成 record_id，断点续跑不重烧 API。
- **usage**：累加 teacher 的 prompt/completion token，即成本账。
- **单样本容错**：单条失败（高德超时/DeepSeek 限流）写 `errors.jsonl` 跳过，不炸整批；`context_fail`/`teacher_fail` 分开记。
- **泄漏 guard**：改为排除**他的 500 条评测请求**（旧版排除的是我们旧 44/39）；防 test-set contamination。

teacher = DeepSeek，长超时（300s，生成完整行程慢，勿用短链）。

### 组件 4：导出（改 `export_llamafactory`）

读 grounded 记录，用 `validate_grounded_trip_plan` 二次确认干净，导出 sharegpt：`system` = `PLANNER_AGENT_PROMPT`，`human` = `build_grounded_planner_messages(compact)[1]`，`gpt` = teacher 的 TripPlan JSON。train/val 切分沿用旧比例（约 95/5）。

## 数据流

```
requestgen(控制变量) → DataGenBuilder.collect(高德结构化 + 历史天气) → compact
  → teacher(DeepSeek) → TripPlan → validate_grounded_trip_plan
  → 无违规? 留 : 丢(errors.jsonl)
  → runs/<slug>/records.jsonl → export → train.json / val.json
```

## 执行节奏

`smoke20 → 人工审 → 100 → 人工审 → ~1200 生（→ ~800-1000 干净）`。每档人工看质量再放大。长跑一律 `nohup + caffeinate + disown`（本地 heavy 任务会被 harness 回收，见 memory）。seed 必须与评测集 seed 区分。

## 测试

- **请求生成器**：同 seed 确定性可复现；控制变量分布符合预期。
- **历史天气 override**：mock open-meteo，断言远期日期拿到非空历史天气、近期仍走高德。
- **泄漏 guard**：构造与评测请求同指纹的请求，断言被跳过。
- **导出格式**：sharegpt 三字段齐全、system 等于 `PLANNER_AGENT_PROMPT`、human 含候选店名。
- **集成**：smoke run（真高德 + 真 DeepSeek，小量）当端到端验证。

## 外部依赖

- **open-meteo 历史档案 API**：免费、无 key，但需网络 + 限流。是 grounded 训练天气的唯一新增外部依赖。

## 风险

- **grounded 数据仍有缺陷**：smoke20 人工审计是第一道闸，先小量发现问题再放大，避免烧一整批 API。
- **AMAP/DeepSeek 成本与时长**：~1200 条约数小时、数 M token；usage 台账实时可查，可随时手刷停。
- **净产率**：旧经验约 59% 干净（约 8% 取数失败 + meal-repeat 等被滤）；grounded 校验更严，产率可能更低，故生 ~1200 目标 ~800-1000。
