# 意图识别模型微调 设计文档

**技术栈：** Python 3.11 / PyTorch / HuggingFace Transformers + Datasets / FastAPI / LangGraph

**Goal:** 用一个自行微调的中文文本分类模型（BERT）取代 `classify_intent` 中的 LLM 分类层，实现 5 类细粒度、本地毫秒级、零 API 成本的意图路由；保留正则快路径与低置信度 LLM 兜底作为安全网。

---

## 背景与问题

当前 `backend/app/agents/intent_classifier.py` 采用「正则规则层 + LLM 兜底」两层结构，只区分 3 类意图（`query_plan` / `modify` / `other`）。存在两个不足：

1. **粒度粗**：`query_plan` 把查天气、查景点、查酒店等揉成一类，下游 `query_handler` 需要再解析一次才能定位 state 字段。
2. **依赖 LLM 分类**：规则未命中即调用 LLM，有 API 成本与网络延迟，且行为不完全可控。

本设计引入一个微调的意图分类模型，做更细的 5 类划分，本地推理替代 LLM 分类层。

---

## 意图体系（5 类）

| 意图 | 含义 | 路由目标 | state 字段 |
|------|------|----------|-----------|
| `query_weather` | 查天气、温度、气温 | `query_handler` | `weather_info` |
| `query_attraction` | 查景点、游玩、餐饮 | `query_handler` | `days[].attractions` |
| `query_hotel` | 查酒店、住宿、预算 | `query_handler` | `days[].hotel` |
| `plan_change` | 生成或修改行程（create 与 modify 合并） | `modify_handler` | 下游按 state 有无 `trip_plan` 决定全量/增量 |
| `other` | 闲聊、问候、域外无关内容 | `other_handler` | — |

**设计取舍：**
- **create 与 modify 合并为 `plan_change`**：二者在意图层不需区分。全量规划（SupervisorGraph）与增量修改（modify_handler）的选择由下游根据 state 中是否已存在 `trip_plan` 决定，而非由意图模型判断。
- **查询按领域拆为 3 类**：因为 `query_handler` 需从不同 state 字段取数，细分后可直接路由，省去二次解析。
- **不再细拆预算/餐饮**：避免样本稀少与类别不均衡；预算并入 `query_hotel`、餐饮并入 `query_attraction`。

---

## 系统架构

### 数据 Pipeline（LLM 合成）

- **脚本** `backend/ml/intent/data_gen.py`
  - 调用 DeepSeek/GPT（复用现有 LLM 配置），按每类定义用 few-shot 提示批量生成多样化中文句子。
  - 规模：**约 300 条/类，合计 ~1500 条**训练集。
  - 输出 `backend/ml/intent/data/train.jsonl`，每行 `{"text": "...", "label": "query_weather"}`。
  - 生成后做去重 + 简单人工抽查清洗。
- **人工测试集** `backend/ml/intent/eval.jsonl`
  - 手写 **~25 条/类真实口吻，合计 ~125 条**。
  - 绝不与训练集混用，仅用于最终评估。纳入 git（体积小、是评估基准）。

### 模型与训练

- **基座模型**：`hfl/chinese-roberta-wwm-ext`（中文表现强、体积适中），外加 5 分类头。
- **框架**：HuggingFace `Trainer`。
- **超参**：约 3 epoch、lr 2e-5、batch 16、max_len 64（可在脚本内常量调整）。
- **脚本** `backend/ml/intent/train.py`
  - 读取 `train.jsonl` → 微调 → 保存到 `backend/models/intent_classifier/`。
  - 训练后在测试集上打印分类报告（precision / recall / F1 per class）。
- **模型产物**：`backend/models/intent_classifier/` 加入 `.gitignore`，不进仓库；仓库只保留脚本与测试集，供他人复现。

### 推理与接入

改造 `backend/app/agents/intent_classifier.py`，分类逻辑为三层：

1. **正则快路径**（保留）：命中高频固定说法（如「谢谢」→ `other`）直接返回，不加载模型。
2. **微调模型本地推理**：正则未命中 → 用 `transformers` pipeline 进程内**单例加载**微调模型，输出 5 类中的一类及 softmax 置信度。
3. **低置信度 LLM 兜底**：置信度 `< 0.7` → 调用现有 `get_agent_llm("intent_classifier")` 做 LLM 分类；LLM 亦失败/不确定时默认返回 `plan_change`（沿用「宁可多判、不误删」的安全偏向）。

**路由改造** `backend/app/agents/chat_graph.py`：
- 意图从 3 类扩到 5 类。
- 3 个 query 类 → `query_handler`，并携带字段提示（weather / attraction / hotel）。
- `plan_change` → `modify_handler`。
- `other` → `other_handler`。
- `query_handler` 增加按意图取对应 state 字段的逻辑。

---

## 评估与成功标准

- **准确性**（手写测试集 `eval.jsonl`）：macro-F1 ≥ 0.90，且每类 recall ≥ 0.85。
- **延迟**：CPU 上单条推理 < 50ms。
- **回归**：现有 36 个测试保持全绿。
- **新增测试** `backend/tests/test_intent_classifier.py`（扩充）：
  - 规则层命中/未命中
  - 模型加载与推理（mock 模型，避免 CI 加载真实权重）
  - 低置信度触发 LLM 兜底
  - 5 类 → 路由目标映射正确

---

## 文件改动清单

```
backend/
  ml/intent/
    data_gen.py              # 新增：LLM 合成训练数据
    train.py                 # 新增：微调训练 + 评估报告
    eval.jsonl               # 新增：手写测试集（进 git）
    data/train.jsonl         # 生成产物（.gitignore）
  models/intent_classifier/  # 模型产物（.gitignore）
  app/agents/intent_classifier.py   # 改造：接入微调模型 + 三层分类
  app/agents/chat_graph.py          # 改造：5 类路由
  requirements.txt                  # 新增 torch / transformers / datasets
  tests/test_intent_classifier.py   # 扩充测试
  .gitignore                        # 忽略 models/ 与 ml/intent/data/
```

**依赖影响**：引入 `torch` + `transformers` + `datasets`，包体较大，会增加 `requirements.txt` 体积与 CI 安装时间。缓解：单元测试 mock 模型加载，不在 CI 中加载真实权重；真实训练与推理仅在本地/训练环境执行。

---

## 错误处理

- 模型文件缺失/加载失败 → 记录告警并**降级为「正则 + LLM」原有两层逻辑**，服务不中断。
- 模型推理置信度低 → 走 LLM 兜底（见上）。
- LLM 兜底失败（网络/超时）→ 默认 `plan_change`，不抛错。

---

## 可扩展性 / 未来工作

后续计划微调本地大模型用于生成更优的行程。为此现在**有意**确立两条约定，但**不提前构建通用框架**（遵循 YAGNI）：

1. **目录约定**：微调任务一律置于 `backend/ml/<task>/`，各自独立（`intent/` 现有，未来 `planner/`）。两任务训练范式差异大（encoder 分类 vs decoder LLM + LoRA），不共用基类；待第二个任务落地后，若确有重复逻辑（如合成数据的 LLM 调用）再抽取到 `ml/common/`。
2. **生成模型接入点**：未来的本地生成模型通过现有 `agents_config.yaml` + `llm_router` 抽象接入——新增一个 provider/模型条目，并把 `assembler` / `modify_handler` 指向它，**无需改动图结构或业务逻辑**。

---

## 非目标（Out of Scope）

- 不微调行程生成模型（本设计仅意图分类）。
- 不构建通用微调框架。
- 不引入独立推理服务（进程内加载即可）。
- 不做交易/转人工等商业产品级意图。
