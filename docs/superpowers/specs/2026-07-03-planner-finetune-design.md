# 行程生成模型微调（Planner SFT）设计文档

**技术栈：** Python 3.11 / LangGraph / FastAPI / LLaMA-Factory（LoRA）/ vLLM / Qwen2.5-7B-Instruct

**Goal:** 用一个 LoRA 微调的本地 7B 生成模型替代 `assembler_node` 中的 DeepSeek API，稳定产出符合 TripPlan 协议的行程 JSON；参照 helloagents-trip-planner 的后训练主线（协议 → 冻结评测 → teacher 数据 → SFT），并保留 DeepSeek 兜底。

---

## 背景：本仓库的两次微调有什么不同

本仓库已有一次微调（2026-07-02 意图分类器，BERT 5 类）。本次 Planner 微调与它是**完全不同的训练范式**，参照对象是 `helloagents-trip-planner/training/` 的后训练实战：

| 维度 | 意图分类微调（已有） | Planner 生成微调（本设计） |
|------|--------------------|--------------------------|
| 模型类型 | encoder 分类（chinese-roberta, ~100M） | decoder 生成（Qwen2.5-7B-Instruct + LoRA） |
| 输出 | 5 类标签 | 数千 token 的长 TripPlan JSON |
| 数据 | LLM 合成短句 ~1500 条 + 手写测试集 125 条 | teacher（DeepSeek）生成完整行程轨迹，**逐条规则审计**后入库；smoke 20 → 100 → 1000 递进 |
| 评测 | macro-F1 / recall | 冻结 standard/hard 评测集 + 规则评测（hardpass/softpass、预算重算） |
| 训练前提 | 无需改业务协议 | **必须先做协议改造**：结构化 party/预算、PlannerContext、输出硬校验（helloagents 的核心教训：输入协议不干净，训练只会放大问题） |
| 训练资源 | CPU/笔记本数分钟 | 1×24GB GPU 可跑（8k ctx 贴边），1×40GB 更稳 |
| 部署 | 进程内 transformers pipeline | vLLM 独立服务，OpenAI-compatible 接入 `llm_router`（沿用意图设计文档预留的接入点） |
| 依赖影响 | backend 引入 torch/transformers | **backend 零新增运行时依赖**（训练在独立 LLaMA-Factory 环境，推理走 HTTP） |
| 失败兜底 | 正则 → 模型 → LLM 三层 | 本地模型 → DeepSeek 链路兜底（`llm_router` 现有 fallback 机制） |

从 helloagents 吸收、且比它省事的地方：

1. **本仓库子图已输出结构化 pydantic 对象**（`weather_outputs/hotel_outputs/poi_outputs`），不像 helloagents 早期要从 agent 原始文本里重新编译，PlannerContext 构建大幅简化。
2. **从第一天就带 manifest + token usage**：helloagents 是中途才补上这条纪律；意图微调的 `data_gen.py` 也没有留 manifest，本设计所有数据 run 强制写入独立 run 目录并记录 usage（此纪律也建议回补到意图数据脚本，不在本设计范围内）。
3. **v1 只做 SFT，不做 DPO / Best-of-N / rerank**：helloagents 的结论是 SFT 学协议收益最大，偏好训练在 SFT 收益变钝后才有意义。规模也缩小：评测集 50+50（对方 200+300）、首轮训练数据 ~1000 条。

必须先修的历史问题：当前 `assembler_node` 的 prompt 里仍有 `"distance": "距离景点2公里"` 伪精确示例（helloagents 教程 3.3 节点名要删的"半可信信号"），餐饮示例是"早餐推荐"这类占位词，且 `agents_config.yaml` 里配置的 `assembler` 条目实际未被使用（代码直接走 `acall_with_fallback` 全局链）。

---

## 系统架构

### 阶段 A：协议改造（无 GPU）

- `schemas.py` 增加 `PartyInfo`（adults/children/elders，computed total）与 `BudgetConstraint`（amount/scope/budget_level/strictness），`TripRequest` 以**带默认值的可选字段**接入，前端不改也兼容。
- 新增 `app/planner/context.py`：把 `TripRequest` + 三个子图的结构化输出编译成 `PlannerContext` dict（request/party/budget_constraint/lodging_policy/pricing_policy/tool_snapshot/planner_constraints），并生成 Planner prompt。**训练、评测、推理共用这一个来源**（对应意图微调 `intent_labels.py` 的单一源经验）。
- 新增 `app/planner/validation.py`：TripPlan shape 校验（日期/天数/三餐/住宿日 hotel 非空/餐饮占位词/`hotel.distance` 必须为空/候选 grounding/天气复制）+ `recompute_budget()` 工程重算预算（酒店按晚、门票×人数、餐饮×人数）。
- `assembler_node` 改为：PlannerContext prompt → `acall_agent_with_fallback("assembler", ...)`（新增：优先用 `agents_config.yaml` 指定的模型，异常降级到全局 DeepSeek 链）→ 解析 → 校验告警 → 预算重算。

### 阶段 B：冻结评测集与规则评测（无 GPU）

- `ml/planner/requestgen.py`：种子可复现的受控请求生成器，standard（2-4 天、1-2 人、宽松预算）与 hard（4-6 天、多人/儿童老人、hard 紧预算、负向约束）两档；提供 `eval_signature()` 供训练数据防泄漏。
- `ml/planner/build_eval_set.py`：真跑三个子图给每条请求拍快照，产出冻结评测集 `ml/planner/eval/records.jsonl`（standard 50）与 `ml/planner/eval_hard/records.jsonl`（hard 50），**入 git、建成后不再重采样**（检索逻辑变了只重建 context，保留 record_id）。
- `ml/planner/rule_eval.py`：对任意 OpenAI-compatible 端点跑评测，输出 hardpass（JSON/schema/日期/三餐/住宿/grounding/占位词）与 softpass（餐饮不重复、预算落区间），生成 generations.jsonl + report。

### 阶段 C：SFT 数据（teacher = DeepSeek，无 GPU）

- `ml/planner/data_gen.py`：请求（与评测集不同 seed + 签名过滤）→ 子图快照 → teacher 生成 → 规则硬过滤，通过进 `records.jsonl`、失败进 `errors.jsonl`；每个 run 独立目录 `ml/planner/data/runs/<YYMMDD>_<slug>/`，`manifest.json` 记录通过率与 token usage。节奏：smoke 20 → 审计 → 100 → 审计 → 1000。
- `ml/planner/export_llamafactory.py`：导出 sharegpt 格式 train/val + `dataset_info.json`。

### 阶段 D：训练与接入（GPU）

- LLaMA-Factory LoRA：r=32 / alpha=64 / lora_target=all / cutoff_len=8192（本仓库 context 远短于 helloagents 的 24k）/ lr 5e-5 / 3 epoch / bf16 + gradient checkpointing。
- 合并 LoRA → vLLM 服务（`--served-model-name trip-planner-sft`）→ `config.py` 增加 `local_*` 三项 → `agents_config.yaml` 把 `assembler` 指向 `local`（完全复用 `get_agent_llm` 的 provider 机制，不改图结构）。

---

## 评估与成功标准

在冻结评测集上对比三方：DeepSeek 基线 / Qwen2.5-7B 未微调 / SFT 模型。

- **hardpass**：standard ≥ 85%，hard ≥ 70%，且比未微调基座高 ≥ 20pp。
- **不回退**：SFT 模型 hardpass 不低于 DeepSeek 基线 −5pp（低于则继续加数据迭代，不切换线上）。
- **回归**：现有测试全绿（意图微调计划落地后为其全量，未落地则为现 36 个）。
- 达标后才把 `agents_config.yaml` 的 `assembler` 切到 `local`；DeepSeek 兜底常驻。

## 错误处理

- 本地模型服务不可达/超时/输出坏 JSON → `acall_agent_with_fallback` 降级 DeepSeek 全局链，服务不中断。
- TripPlan 校验失败 → 线上只告警不拦截（violations 记日志），预算一律工程重算；训练数据侧则硬过滤。

## 非目标（Out of Scope）

- 不做 DPO / Best-of-N / 多候选 rerank（SFT 收益变钝后另立计划）。
- 不微调 modify_handler / 对话链路（接入点相同，后续切配置即可）。
- 不建票价表、不接路线 API（`ticket_price` 用候选解析值，`distance` 一律留空）。
- 前端表单不强制提交 party/预算（schema 已就绪，另行改版）。
