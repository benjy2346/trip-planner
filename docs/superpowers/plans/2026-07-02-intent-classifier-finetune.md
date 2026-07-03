# 意图识别模型微调 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一个微调的中文 BERT 分类模型（5 类）取代 `classify_intent` 的 LLM 分类层，本地推理路由对话意图，保留正则快路径与低置信度 LLM 兜底。

**Architecture:** 单一标签源 `intent_labels.py` 供训练与推理共用；`ml/intent/` 下的合成数据脚本与训练脚本产出模型到 `models/intent_classifier/`；`intent_model.py` 进程内单例加载模型做推理；`intent_classifier.py` 三层分类（正则 → 模型 → LLM 兜底）；`chat_graph.py` 按 5 类路由。

**Tech Stack:** Python 3.11 / PyTorch / HuggingFace Transformers + Datasets / scikit-learn / LangGraph / FastAPI

## Global Constraints

- Python 3.11。
- 现有 36 个测试必须保持全绿。
- 5 个意图标签固定顺序：`["query_weather", "query_attraction", "query_hotel", "plan_change", "other"]`。
- 基座模型：`hfl/chinese-roberta-wwm-ext`。
- 置信度阈值 0.7；模型/LLM 均不确定时默认 `plan_change`。
- 模型产物目录 `backend/models/` 与合成数据 `backend/ml/intent/data/` 不进 git；`eval.jsonl` 进 git。
- 成功标准：手写测试集上 macro-F1 ≥ 0.90，每类 recall ≥ 0.85；CPU 单条推理 < 50ms。
- 单元测试一律 mock 模型加载，不在 CI 加载真实权重。

---

## File Structure

```
backend/
  app/agents/
    intent_labels.py      # 新增：5 类标签 + id 映射 + 路由映射（单一源）
    intent_model.py       # 新增：微调模型进程内单例加载 + predict()
    intent_classifier.py  # 改造：三层分类（正则 → 模型 → LLM）
    chat_graph.py         # 改造：5 类路由 + intent 提示
    state.py              # 改造：SupervisorState 增加 intent 字段
  ml/intent/
    data_gen.py           # 新增：LLM 合成训练数据
    train.py              # 新增：微调训练 + 评估报告
    eval.jsonl            # 新增：手写测试集（进 git）
    data/train.jsonl      # 生成产物（.gitignore）
  models/intent_classifier/  # 训练产物（.gitignore）
  tests/
    test_intent_labels.py       # 新增
    test_intent_model.py        # 新增
    test_eval_dataset.py        # 新增
    test_intent_classifier.py   # 重写（5 类三层）
    test_chat_graph_nodes.py    # 更新路由测试
  requirements.txt        # 新增 torch/transformers/datasets/scikit-learn/accelerate
  .gitignore              # 新增 models/ 与 ml/intent/data/
```

---

## Task 1: 依赖、gitignore 与标签常量（单一源）

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/.gitignore`
- Create: `backend/app/agents/intent_labels.py`
- Test: `backend/tests/test_intent_labels.py`

**Interfaces:**
- Produces:
  - `INTENT_LABELS: list[str]`（长度 5，固定顺序）
  - `LABEL2ID: dict[str, int]`、`ID2LABEL: dict[int, str]`
  - `INTENT_TO_NODE: dict[str, str]`（意图 → chat_graph 节点名）
  - `QUERY_INTENT_FIELD: dict[str, str]`（query 类意图 → state 字段名 weather/attraction/hotel）

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_intent_labels.py
from app.agents.intent_labels import (
    INTENT_LABELS, LABEL2ID, ID2LABEL, INTENT_TO_NODE, QUERY_INTENT_FIELD,
)


def test_five_labels_fixed_order():
    assert INTENT_LABELS == [
        "query_weather", "query_attraction", "query_hotel", "plan_change", "other",
    ]


def test_label_id_roundtrip():
    for i, label in enumerate(INTENT_LABELS):
        assert LABEL2ID[label] == i
        assert ID2LABEL[i] == label


def test_every_label_routes_to_a_node():
    for label in INTENT_LABELS:
        assert INTENT_TO_NODE[label] in {"query_handler", "modify_handler", "other_handler"}


def test_query_intents_map_to_fields():
    assert QUERY_INTENT_FIELD == {
        "query_weather": "weather",
        "query_attraction": "attraction",
        "query_hotel": "hotel",
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_intent_labels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.agents.intent_labels'`

- [ ] **Step 3: 创建标签模块**

```python
# backend/app/agents/intent_labels.py
"""意图标签与路由映射（训练与推理共用的单一来源）"""

INTENT_LABELS: list[str] = [
    "query_weather",
    "query_attraction",
    "query_hotel",
    "plan_change",
    "other",
]

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

INTENT_TO_NODE: dict[str, str] = {
    "query_weather": "query_handler",
    "query_attraction": "query_handler",
    "query_hotel": "query_handler",
    "plan_change": "modify_handler",
    "other": "other_handler",
}

QUERY_INTENT_FIELD: dict[str, str] = {
    "query_weather": "weather",
    "query_attraction": "attraction",
    "query_hotel": "hotel",
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_intent_labels.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: 更新 requirements.txt**

在 `backend/requirements.txt` 末尾「# 测试」段之前，追加一段：

```
# 意图分类模型（微调）
torch>=2.2.0
transformers>=4.40.0
datasets>=2.19.0
scikit-learn>=1.4.0
accelerate>=0.30.0
```

- [ ] **Step 6: 更新 .gitignore**

在 `backend/.gitignore` 末尾追加：

```
# 微调模型产物与合成数据
models/
ml/intent/data/
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/agents/intent_labels.py backend/tests/test_intent_labels.py \
        backend/requirements.txt backend/.gitignore
git commit -m "feat: add intent label constants, ML deps, and gitignore for finetune"
```

---

## Task 2: 手写测试集 eval.jsonl

**Files:**
- Create: `backend/ml/intent/eval.jsonl`
- Test: `backend/tests/test_eval_dataset.py`

**Interfaces:**
- Consumes: `INTENT_LABELS`（Task 1）
- Produces: `backend/ml/intent/eval.jsonl`，每行 `{"text": str, "label": str}`，每类 ≥ 20 条。

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_eval_dataset.py
import json
from collections import Counter
from pathlib import Path
from app.agents.intent_labels import INTENT_LABELS

EVAL_PATH = Path(__file__).resolve().parent.parent / "ml" / "intent" / "eval.jsonl"


def _load():
    rows = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_eval_exists_and_wellformed():
    rows = _load()
    assert len(rows) >= 100
    for r in rows:
        assert set(r.keys()) == {"text", "label"}
        assert isinstance(r["text"], str) and r["text"].strip()
        assert r["label"] in INTENT_LABELS


def test_eval_min_per_class():
    counts = Counter(r["label"] for r in _load())
    for label in INTENT_LABELS:
        assert counts[label] >= 20, f"{label} only has {counts[label]} eval rows"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_eval_dataset.py -q`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: 创建 eval.jsonl（每类 ≥ 20 条，以下为种子，按同风格扩写到 25/类）**

在 `backend/ml/intent/eval.jsonl` 写入手写样本。以下每类给出 10 条种子，**请按相同口吻各扩写到 25 条左右**（真实、口语化、避免与训练集雷同）：

```jsonl
{"text": "那几天天气怎么样", "label": "query_weather"}
{"text": "会不会下雨啊", "label": "query_weather"}
{"text": "第二天冷不冷", "label": "query_weather"}
{"text": "白天气温大概多少度", "label": "query_weather"}
{"text": "需要带外套吗", "label": "query_weather"}
{"text": "出发那天热不热", "label": "query_weather"}
{"text": "晚上温度低吗", "label": "query_weather"}
{"text": "风大不大", "label": "query_weather"}
{"text": "这几天是晴天吗", "label": "query_weather"}
{"text": "天气预报怎么说", "label": "query_weather"}
{"text": "第一天去哪些景点", "label": "query_attraction"}
{"text": "有什么好玩的地方", "label": "query_attraction"}
{"text": "中午吃什么", "label": "query_attraction"}
{"text": "第三天的游览安排是啥", "label": "query_attraction"}
{"text": "推荐的餐厅是哪家", "label": "query_attraction"}
{"text": "故宫要玩多久", "label": "query_attraction"}
{"text": "晚餐吃啥", "label": "query_attraction"}
{"text": "有没有适合拍照的景点", "label": "query_attraction"}
{"text": "第二天参观什么", "label": "query_attraction"}
{"text": "门票多少钱一张", "label": "query_attraction"}
{"text": "住哪个酒店", "label": "query_hotel"}
{"text": "第一天住哪儿", "label": "query_hotel"}
{"text": "酒店离景点远吗", "label": "query_hotel"}
{"text": "住宿一晚多少钱", "label": "query_hotel"}
{"text": "总共要花多少钱", "label": "query_hotel"}
{"text": "预算大概是多少", "label": "query_hotel"}
{"text": "住的地方叫什么名字", "label": "query_hotel"}
{"text": "酒店评分高不高", "label": "query_hotel"}
{"text": "这次旅行总费用是多少", "label": "query_hotel"}
{"text": "住宿是什么档次的", "label": "query_hotel"}
{"text": "帮我把第二天改得轻松一点", "label": "plan_change"}
{"text": "删掉第三天的博物馆", "label": "plan_change"}
{"text": "我想多加一天", "label": "plan_change"}
{"text": "把行程改成五天", "label": "plan_change"}
{"text": "换一个便宜点的酒店", "label": "plan_change"}
{"text": "第一天不想去故宫了", "label": "plan_change"}
{"text": "帮我重新规划一下", "label": "plan_change"}
{"text": "加个爬山的行程", "label": "plan_change"}
{"text": "把午餐换成川菜", "label": "plan_change"}
{"text": "行程太赶了，调松一些", "label": "plan_change"}
{"text": "谢谢你", "label": "other"}
{"text": "你好", "label": "other"}
{"text": "好的没问题", "label": "other"}
{"text": "你是谁", "label": "other"}
{"text": "今天股市怎么样", "label": "other"}
{"text": "讲个笑话吧", "label": "other"}
{"text": "辛苦了", "label": "other"}
{"text": "嗯嗯收到", "label": "other"}
{"text": "你能做什么", "label": "other"}
{"text": "帮我写首诗", "label": "other"}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_eval_dataset.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/ml/intent/eval.jsonl backend/tests/test_eval_dataset.py
git commit -m "test: add hand-written intent eval dataset with format checks"
```

---

## Task 3: 合成训练数据脚本 data_gen.py

**Files:**
- Create: `backend/ml/intent/data_gen.py`
- Test: `backend/tests/test_data_gen.py`

**Interfaces:**
- Consumes: `INTENT_LABELS`（Task 1）；现有 `app.agents.llm_router.acall_with_fallback`
- Produces:
  - `dedup(rows: list[dict]) -> list[dict]`（按 text 去重，保序）
  - `build_prompt(label: str, n: int) -> str`
  - CLI 入口：运行后写 `backend/ml/intent/data/train.jsonl`

- [ ] **Step 1: 写失败测试（纯函数部分）**

```python
# backend/tests/test_data_gen.py
from app.agents.intent_labels import INTENT_LABELS
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "data_gen",
    Path(__file__).resolve().parent.parent / "ml" / "intent" / "data_gen.py",
)
data_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_gen)


def test_dedup_removes_duplicates_preserving_order():
    rows = [
        {"text": "a", "label": "other"},
        {"text": "b", "label": "other"},
        {"text": "a", "label": "other"},
    ]
    out = data_gen.dedup(rows)
    assert [r["text"] for r in out] == ["a", "b"]


def test_build_prompt_mentions_label_and_count():
    p = data_gen.build_prompt("query_weather", 50)
    assert "query_weather" in p
    assert "50" in p
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_data_gen.py -q`
Expected: FAIL (无法加载 data_gen.py)

- [ ] **Step 3: 创建 data_gen.py**

```python
# backend/ml/intent/data_gen.py
"""用 LLM 合成意图分类训练数据。

运行：cd backend && python -m ml.intent.data_gen
输出：backend/ml/intent/data/train.jsonl
"""
import asyncio
import json
from pathlib import Path

from app.agents.intent_labels import INTENT_LABELS
from app.agents.llm_router import acall_with_fallback
from langchain_core.messages import SystemMessage, HumanMessage

PER_LABEL = 300
OUT_PATH = Path(__file__).resolve().parent / "data" / "train.jsonl"

_LABEL_DESC = {
    "query_weather": "查询行程期间的天气、气温、是否下雨、穿衣建议",
    "query_attraction": "查询景点、游玩安排、游览时长、餐饮/吃饭推荐",
    "query_hotel": "查询住宿/酒店信息、酒店位置评分、住宿费用与总预算",
    "plan_change": "生成新行程，或修改、增删、调整已有行程的任何内容",
    "other": "问候、感谢、闲聊，或与旅行行程完全无关的内容",
}


def build_prompt(label: str, n: int) -> str:
    return (
        f"你在为一个中文旅行助手构造意图分类训练数据。\n"
        f"意图类别「{label}」的含义：{_LABEL_DESC[label]}。\n"
        f"请生成 {n} 条属于该意图的、多样化的中文用户消息，"
        f"口语化、长短不一、涵盖不同问法，避免重复。\n"
        f"每行一条，只输出消息文本，不要编号、不要引号、不要多余说明。"
    )


def dedup(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r)
    return out


async def _gen_label(label: str, n: int) -> list[dict]:
    resp = await acall_with_fallback([
        SystemMessage(content="你是数据标注助手，只输出要求的内容。"),
        HumanMessage(content=build_prompt(label, n)),
    ])
    rows = []
    for line in resp.content.splitlines():
        text = line.strip().strip("\"'　 ")
        if text:
            rows.append({"text": text, "label": label})
    return rows


async def main() -> None:
    all_rows: list[dict] = []
    for label in INTENT_LABELS:
        rows = await _gen_label(label, PER_LABEL)
        print(f"{label}: 生成 {len(rows)} 条")
        all_rows.extend(rows)
    all_rows = dedup(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {len(all_rows)} 条到 {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_data_gen.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: 真跑一次生成数据（需要 LLM API key）**

Run: `cd backend && python3 -m ml.intent.data_gen`
Expected: 打印每类生成条数，最终 `写入 N 条到 .../data/train.jsonl`（N 约 1200-1500）。
校验：`wc -l ml/intent/data/train.jsonl`，并抽查若干行标签是否合理，明显错标的手动删除。

- [ ] **Step 6: Commit（注意 train.jsonl 已被 gitignore，不会提交）**

```bash
git add backend/ml/intent/data_gen.py backend/tests/test_data_gen.py
git commit -m "feat: add LLM synthetic data generation for intent classifier"
```

---

## Task 4: 微调训练脚本 train.py

**Files:**
- Create: `backend/ml/intent/train.py`
- Test: `backend/tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `INTENT_LABELS`, `LABEL2ID`, `ID2LABEL`（Task 1）；`data/train.jsonl`（Task 3）；`eval.jsonl`（Task 2）
- Produces:
  - `load_jsonl(path) -> list[dict]`
  - `compute_metrics(eval_pred) -> dict`（含 `macro_f1`）
  - 训练产物目录 `backend/models/intent_classifier/`

- [ ] **Step 1: 写失败测试（纯函数部分）**

```python
# backend/tests/test_train_helpers.py
import json
import importlib.util
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "train",
    Path(__file__).resolve().parent.parent / "ml" / "intent" / "train.py",
)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def test_load_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"text":"a","label":"other"}\n\n{"text":"b","label":"query_hotel"}\n',
        encoding="utf-8",
    )
    rows = train.load_jsonl(p)
    assert rows == [
        {"text": "a", "label": "other"},
        {"text": "b", "label": "query_hotel"},
    ]


def test_compute_metrics_perfect():
    logits = np.array([[9.0, 0, 0, 0, 0], [0, 0, 0, 0, 9.0]])
    labels = np.array([0, 4])
    m = train.compute_metrics((logits, labels))
    assert m["macro_f1"] == 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_train_helpers.py -q`
Expected: FAIL (无法加载 train.py)

- [ ] **Step 3: 创建 train.py**

```python
# backend/ml/intent/train.py
"""微调中文 BERT 意图分类器。

运行：cd backend && python -m ml.intent.train
产物：backend/models/intent_classifier/
"""
import json
from pathlib import Path

import numpy as np

from app.agents.intent_labels import INTENT_LABELS, LABEL2ID, ID2LABEL

BASE_MODEL = "hfl/chinese-roberta-wwm-ext"
HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "data" / "train.jsonl"
EVAL_PATH = HERE / "eval.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "intent_classifier"

MAX_LEN = 64
EPOCHS = 3
LR = 2e-5
BATCH = 16


def load_jsonl(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, recall_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_recall": recall_score(labels, preds, average="macro"),
    }


def main() -> None:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments,
    )
    from sklearn.metrics import classification_report

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def to_ds(rows):
        return Dataset.from_dict({
            "text": [r["text"] for r in rows],
            "label": [LABEL2ID[r["label"]] for r in rows],
        })

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = to_ds(load_jsonl(TRAIN_PATH)).map(tok, batched=True)
    eval_rows = load_jsonl(EVAL_PATH)
    eval_ds = to_ds(eval_rows).map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "_checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    preds = np.argmax(trainer.predict(eval_ds).predictions, axis=-1)
    gold = [LABEL2ID[r["label"]] for r in eval_rows]
    print(classification_report(
        gold, preds, target_names=INTENT_LABELS, digits=3,
    ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"模型已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_train_helpers.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: 真跑训练（本地，需先完成 Task 3 生成 train.jsonl；有 GPU 更快，CPU 亦可，数分钟级）**

Run: `cd backend && python3 -m ml.intent.train`
Expected: 训练完成后打印 classification_report；**验收标准：`macro avg` 的 f1 ≥ 0.90，且每类 recall ≥ 0.85**。
若不达标：回到 Task 3 增加该类数据多样性/数量，或提高 EPOCHS，重训。
确认 `backend/models/intent_classifier/` 下生成了 `config.json` / `model.safetensors` / tokenizer 文件。

- [ ] **Step 6: Commit（模型产物已被 gitignore）**

```bash
git add backend/ml/intent/train.py backend/tests/test_train_helpers.py
git commit -m "feat: add BERT finetune training script for intent classifier"
```

---

## Task 5: 推理封装 intent_model.py

**Files:**
- Create: `backend/app/agents/intent_model.py`
- Test: `backend/tests/test_intent_model.py`

**Interfaces:**
- Produces:
  - `MODEL_DIR: Path`
  - `class IntentModelUnavailable(Exception)`
  - `predict(text: str) -> tuple[str, float]`（返回 (label, confidence)）
  - `get_pipeline()`、`reset()`（单例管理，供测试重置）

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/test_intent_model.py
from unittest.mock import patch, MagicMock
import pytest
from app.agents import intent_model


def teardown_function():
    intent_model.reset()


def test_predict_returns_label_and_confidence():
    fake_pipe = MagicMock(return_value=[[
        {"label": "query_weather", "score": 0.1},
        {"label": "query_hotel", "score": 0.82},
        {"label": "other", "score": 0.08},
    ]])
    with patch("app.agents.intent_model.get_pipeline", return_value=fake_pipe):
        label, conf = intent_model.predict("住哪个酒店")
    assert label == "query_hotel"
    assert conf == pytest.approx(0.82)


def test_missing_model_dir_raises_unavailable():
    intent_model.reset()
    with patch.object(intent_model.MODEL_DIR, "exists", return_value=False):
        with pytest.raises(intent_model.IntentModelUnavailable):
            intent_model.get_pipeline()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_intent_model.py -q`
Expected: FAIL (无 intent_model)

- [ ] **Step 3: 创建 intent_model.py**

```python
# backend/app/agents/intent_model.py
"""微调意图分类模型的进程内单例推理封装。"""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "intent_classifier"

_pipeline = None


class IntentModelUnavailable(Exception):
    """模型目录不存在或加载失败。"""


def _load_pipeline():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        top_k=None,
    )


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_DIR.exists():
            raise IntentModelUnavailable(f"model dir not found: {MODEL_DIR}")
        _pipeline = _load_pipeline()
    return _pipeline


def predict(text: str) -> tuple[str, float]:
    scores = get_pipeline()(text)[0]
    best = max(scores, key=lambda x: x["score"])
    return best["label"], float(best["score"])


def reset() -> None:
    """清空单例（测试用）。"""
    global _pipeline
    _pipeline = None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_intent_model.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/intent_model.py backend/tests/test_intent_model.py
git commit -m "feat: add in-process singleton inference wrapper for intent model"
```

---

## Task 6: 改造 intent_classifier.py 为 5 类三层分类

**Files:**
- Modify: `backend/app/agents/intent_classifier.py`（整文件替换）
- Test: `backend/tests/test_intent_classifier.py`（整文件替换）

**Interfaces:**
- Consumes: `intent_model.predict` / `IntentModelUnavailable`（Task 5）；`get_agent_llm`（现有）
- Produces:
  - `Intent = Literal["query_weather","query_attraction","query_hotel","plan_change","other"]`
  - `classify_by_rules(message: str) -> Intent | None`
  - `class IntentResult(BaseModel)`（`intent: Intent`, `confidence: float`）
  - `async classify_intent(message: str) -> Intent`

- [ ] **Step 1: 整体替换测试文件**

```python
# backend/tests/test_intent_classifier.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.intent_classifier import classify_by_rules, classify_intent, IntentResult
from app.agents.intent_model import IntentModelUnavailable


# --- 规则层 ---

def test_rule_weather():
    assert classify_by_rules("天气怎么样") == "query_weather"

def test_rule_hotel():
    assert classify_by_rules("第一天住哪个酒店") == "query_hotel"

def test_rule_budget_is_hotel():
    assert classify_by_rules("总费用是多少") == "query_hotel"

def test_rule_attraction():
    assert classify_by_rules("第3天景点有哪些") == "query_attraction"

def test_rule_food_is_attraction():
    assert classify_by_rules("中午吃什么") == "query_attraction"

def test_rule_other_thanks():
    assert classify_by_rules("谢谢") == "other"

def test_rule_no_match_returns_none():
    assert classify_by_rules("帮我把第二天改得轻松一点") is None


# --- classify_intent 三层 ---

@pytest.mark.asyncio
async def test_rule_first_skips_model_and_llm():
    with patch("app.agents.intent_classifier.intent_model.predict") as mp, \
         patch("app.agents.intent_classifier.get_agent_llm") as ml:
        result = await classify_intent("第一天住哪")
    mp.assert_not_called()
    ml.assert_not_called()
    assert result == "query_hotel"


@pytest.mark.asyncio
async def test_model_high_confidence_returned():
    with patch("app.agents.intent_classifier.intent_model.predict",
               return_value=("plan_change", 0.93)), \
         patch("app.agents.intent_classifier.get_agent_llm") as ml:
        result = await classify_intent("删掉第三天的博物馆换成购物")
    ml.assert_not_called()
    assert result == "plan_change"


@pytest.mark.asyncio
async def test_model_low_confidence_falls_back_to_llm():
    mock_result = IntentResult(intent="plan_change", confidence=0.95)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)
    with patch("app.agents.intent_classifier.intent_model.predict",
               return_value=("query_hotel", 0.4)), \
         patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("嗯那个你看着办")
    assert result == "plan_change"


@pytest.mark.asyncio
async def test_model_unavailable_falls_back_to_llm():
    mock_result = IntentResult(intent="query_weather", confidence=0.9)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)
    with patch("app.agents.intent_classifier.intent_model.predict",
               side_effect=IntentModelUnavailable), \
         patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("那几天会不会冷")
    assert result == "query_weather"


@pytest.mark.asyncio
async def test_llm_exception_defaults_to_plan_change():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=Exception("boom"))
    with patch("app.agents.intent_classifier.intent_model.predict",
               side_effect=IntentModelUnavailable), \
         patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("随便改改")
    assert result == "plan_change"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_intent_classifier.py -q`
Expected: FAIL（旧实现返回旧标签 / import 报错）

- [ ] **Step 3: 整体替换 intent_classifier.py**

```python
# backend/app/agents/intent_classifier.py
import re
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.llm_router import get_agent_llm
from app.agents import intent_model
from app.agents.intent_model import IntentModelUnavailable

Intent = Literal["query_weather", "query_attraction", "query_hotel", "plan_change", "other"]

CONFIDENCE_THRESHOLD = 0.7

QUERY_WEATHER_RULES: list[str] = [r"(天气|温度|气温|下雨|冷不冷|热不热)"]
QUERY_HOTEL_RULES: list[str] = [
    r"(住哪|酒店|住宿|宾馆)",
    r"(预算|费用|花多少|多少钱|总价)",
]
QUERY_ATTRACTION_RULES: list[str] = [
    r"(景点|去哪|参观|游览|好玩)",
    r"(餐|吃什么|午餐|晚餐|早餐|餐厅)",
]
OTHER_RULES: list[str] = [
    r"^(谢谢|感谢|好的|可以|没问题|好|嗯|收到)[！!。]*$",
    r"^(你好|您好|hi|hello)[！!。]*$",
]

_CLASSIFIER_PROMPT = (
    "你是行程助手的意图分类器。根据用户消息，判断意图：\n"
    "- query_weather：查询天气、气温、是否下雨、穿衣建议\n"
    "- query_attraction：查询景点、游玩安排、游览时长、餐饮/吃饭\n"
    "- query_hotel：查询住宿/酒店、酒店位置评分、住宿费用与总预算\n"
    "- plan_change：生成新行程，或修改、增删、调整已有行程\n"
    "- other：问候、感谢、闲聊或与行程无关的内容\n"
    "返回结构化 JSON，包含 intent 和 confidence（0.0-1.0）。"
)


class IntentResult(BaseModel):
    intent: Intent
    confidence: float


def classify_by_rules(message: str) -> Intent | None:
    for pattern in QUERY_WEATHER_RULES:
        if re.search(pattern, message):
            return "query_weather"
    for pattern in QUERY_HOTEL_RULES:
        if re.search(pattern, message):
            return "query_hotel"
    for pattern in QUERY_ATTRACTION_RULES:
        if re.search(pattern, message):
            return "query_attraction"
    for pattern in OTHER_RULES:
        if re.search(pattern, message, re.IGNORECASE):
            return "other"
    return None


async def _classify_by_llm(message: str) -> Intent:
    try:
        llm = get_agent_llm("intent_classifier")
        structured = llm.with_structured_output(IntentResult, method="function_calling")
        result: IntentResult = await structured.ainvoke([
            SystemMessage(content=_CLASSIFIER_PROMPT),
            HumanMessage(content=message),
        ])
        if result.confidence < CONFIDENCE_THRESHOLD:
            return "plan_change"
        return result.intent
    except Exception:
        return "plan_change"


async def classify_intent(message: str) -> Intent:
    rule_result = classify_by_rules(message)
    if rule_result is not None:
        return rule_result

    try:
        label, confidence = intent_model.predict(message)
        if confidence >= CONFIDENCE_THRESHOLD:
            return label  # type: ignore[return-value]
    except IntentModelUnavailable:
        pass
    except Exception:
        pass

    return await _classify_by_llm(message)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_intent_classifier.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/intent_classifier.py backend/tests/test_intent_classifier.py
git commit -m "feat: 5-class 3-layer intent classification (rules/model/LLM)"
```

---

## Task 7: chat_graph 5 类路由 + query_handler 意图提示

**Files:**
- Modify: `backend/app/agents/state.py`（SupervisorState 增加 intent 字段）
- Modify: `backend/app/agents/chat_graph.py`
- Test: `backend/tests/test_chat_graph_nodes.py`（更新路由测试 + 新增意图提示测试）

**Interfaces:**
- Consumes: `INTENT_TO_NODE`, `QUERY_INTENT_FIELD`（Task 1）；`classify_intent`（Task 6）
- Produces: `classify_intent_node` 通过 `Command(update={"intent": ...})` 写入 state；`query_handler_node` 读取 `state["intent"]` 作字段提示。

- [ ] **Step 1: 更新 state.py，SupervisorState 增加 intent**

将 `backend/app/agents/state.py` 顶部 import 改为：

```python
import operator
from typing import TypedDict, Annotated, Optional, NotRequired
from langgraph.graph.message import add_messages
from app.models.schemas import TripRequest, TripPlan, WeatherInfo, Hotel, Attraction
```

在 `SupervisorState` 的 `poi_outputs` 行后新增一行：

```python
    poi_outputs: Annotated[list, operator.add]
    intent: NotRequired[str]
```

- [ ] **Step 2: 写失败测试（追加到 test_chat_graph_nodes.py 末尾，并修改两处旧路由测试）**

将文件中 `test_classify_intent_node_routes_to_query` 与 `test_classify_intent_node_routes_to_modify` 两个函数整体替换为：

```python
@pytest.mark.asyncio
async def test_classify_intent_node_routes_query_hotel_to_query_handler():
    from unittest.mock import AsyncMock, patch
    from app.agents.chat_graph import classify_intent_node

    with patch("app.agents.chat_graph.classify_intent", AsyncMock(return_value="query_hotel")):
        state = _make_state("第一天住哪")
        cmd = await classify_intent_node(state)

    assert cmd.goto == "query_handler"
    assert cmd.update["intent"] == "query_hotel"


@pytest.mark.asyncio
async def test_classify_intent_node_routes_plan_change_to_modify():
    from unittest.mock import AsyncMock, patch
    from app.agents.chat_graph import classify_intent_node

    with patch("app.agents.chat_graph.classify_intent", AsyncMock(return_value="plan_change")):
        state = _make_state("帮我改一下")
        cmd = await classify_intent_node(state)

    assert cmd.goto == "modify_handler"
    assert cmd.update["intent"] == "plan_change"


@pytest.mark.asyncio
async def test_query_handler_uses_intent_hint_for_weather():
    from app.agents.chat_graph import query_handler_node
    state = _make_state("那几天冷不冷")  # 无「天气」字样，靠 intent 提示
    state["intent"] = "query_weather"
    result = await query_handler_node(state)
    assert "晴" in result["messages"][0].content


@pytest.mark.asyncio
async def test_query_handler_uses_intent_hint_for_attraction():
    from app.agents.chat_graph import query_handler_node
    state = _make_state("有啥安排")
    state["intent"] = "query_attraction"
    result = await query_handler_node(state)
    assert "故宫" in result["messages"][0].content
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cd backend && python3 -m pytest tests/test_chat_graph_nodes.py -q`
Expected: FAIL（旧 `_INTENT_TO_NODE` 无 `query_hotel`/`plan_change` 键；无 intent 提示逻辑）

- [ ] **Step 4: 改造 chat_graph.py**

替换 import 段与 `_INTENT_TO_NODE`（第 1-17 行）为：

```python
import json
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from app.agents.state import SupervisorState
from app.agents.intent_classifier import classify_intent
from app.agents.intent_labels import INTENT_TO_NODE, QUERY_INTENT_FIELD
from app.agents.llm_router import get_agent_llm
from app.models.schemas import TripPlan

_DAY_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
```

新增三个字段汇总辅助函数（放在 `_parse_day` 之后、`_build_query_reply` 之前）：

```python
def _weather_summary(plan: TripPlan) -> str:
    if not plan.weather_info:
        return "暂无天气信息。"
    lines = [
        f"{w.date}：{w.day_weather}，白天 {w.day_temp}°C / 夜间 {w.night_temp}°C"
        for w in plan.weather_info
    ]
    return "天气预报：\n" + "\n".join(lines)


def _hotel_summary(plan: TripPlan) -> str:
    lines = [
        f"第{i}天：{d.hotel.name}（{d.hotel.address}），约 {d.hotel.estimated_cost} 元/晚"
        for i, d in enumerate(plan.days, 1) if d.hotel
    ]
    return "住宿安排：\n" + "\n".join(lines) if lines else "暂无酒店信息。"


def _attraction_summary(plan: TripPlan) -> str:
    lines = [
        f"第{i}天：{a.name}（建议 {a.visit_duration} 分钟）"
        for i, d in enumerate(plan.days, 1) for a in d.attractions
    ]
    return "景点安排：\n" + "\n".join(lines) if lines else "暂无景点信息。"
```

将 `_build_query_reply` 的签名与「天气」分支及结尾兜底改为使用意图提示——整体替换 `_build_query_reply` 为：

```python
def _build_query_reply(message: str, state: SupervisorState, intent: str | None = None) -> str:
    plan: TripPlan | None = state.get("trip_plan")
    if not plan:
        return "还没有生成行程，请先规划行程。"

    if re.search(r"(天气|温度|气温)", message):
        return _weather_summary(plan)

    if re.search(r"(预算|费用|花多少|多少钱)", message):
        b = plan.budget
        if not b:
            return "暂无预算信息。"
        return (
            f"总预算：{b.total} 元\n"
            f"  景点门票：{b.total_attractions} 元\n"
            f"  住宿：{b.total_hotels} 元\n"
            f"  餐饮：{b.total_meals} 元\n"
            f"  交通：{b.total_transportation} 元"
        )

    day = _parse_day(message)
    if day is not None:
        idx = day - 1
        if idx < 0 or idx >= len(plan.days):
            return f"行程只有 {len(plan.days)} 天，没有第 {day} 天。"
        d = plan.days[idx]

        if re.search(r"(住哪|酒店|住宿)", message):
            h = d.hotel
            if not h:
                return "暂无酒店信息。"
            return f"第{day}天住宿：{h.name}（{h.address}），约 {h.estimated_cost} 元/晚。"

        if re.search(r"(餐|吃什么|午餐|晚餐|早餐)", message):
            if not d.meals:
                return "暂无餐饮信息。"
            lines = [f"  {m.type}：{m.name}，约 {m.estimated_cost} 元" for m in d.meals]
            return f"第{day}天餐饮：\n" + "\n".join(lines)

        if re.search(r"(景点|去哪|参观|游览)", message):
            if not d.attractions:
                return "暂无景点信息。"
            lines = [f"  {a.name}（建议 {a.visit_duration} 分钟）：{a.description[:30]}" for a in d.attractions]
            return f"第{day}天景点：\n" + "\n".join(lines)

        return f"第{day}天（{d.date}）：{d.description}"

    field = QUERY_INTENT_FIELD.get(intent or "")
    if field == "weather":
        return _weather_summary(plan)
    if field == "hotel":
        return _hotel_summary(plan)
    if field == "attraction":
        return _attraction_summary(plan)

    return "请问您想了解行程的哪部分？可以询问天气、预算、各天的景点、酒店或餐饮安排。"
```

将 `query_handler_node` 改为读取并传入 intent：

```python
async def query_handler_node(state: SupervisorState) -> dict:
    messages = state.get("messages", [])
    user_message = messages[-1].content if messages else ""
    reply = _build_query_reply(user_message, state, state.get("intent"))
    return {"messages": [AIMessage(content=reply)]}
```

将 `classify_intent_node` 改为写入 intent 到 state，并用共享映射：

```python
async def classify_intent_node(state: SupervisorState) -> Command:
    messages = state.get("messages", [])
    if not messages:
        return Command(goto="other_handler")
    intent = await classify_intent(messages[-1].content)
    return Command(goto=INTENT_TO_NODE[intent], update={"intent": intent})
```

（删除原文件中局部的 `_INTENT_TO_NODE` 定义。）

- [ ] **Step 5: 运行测试确认通过**

Run: `cd backend && python3 -m pytest tests/test_chat_graph_nodes.py -q`
Expected: PASS（原有 6 个 + 新增/改写 4 个全部通过）

- [ ] **Step 6: Commit**

```bash
git add backend/app/agents/state.py backend/app/agents/chat_graph.py backend/tests/test_chat_graph_nodes.py
git commit -m "feat: 5-class chat_graph routing with intent hint in query_handler"
```

---

## Task 8: 全量回归 + README 说明

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: 全部前序任务

- [ ] **Step 1: 跑全量测试确认全绿**

Run: `cd backend && python3 -m pytest tests/ -q`
Expected: PASS（原 36 个 + 新增意图分类相关全部通过，无 failed/error）

- [ ] **Step 2: README 增加「意图分类模型」小节**

在 `README.md` 的「🔧 核心实现」小节末尾追加：

```markdown
### 意图分类模型（微调）

对话意图由一个微调的中文 BERT 分类器（`hfl/chinese-roberta-wwm-ext`，5 类）识别，
取代原 LLM 分类层。分类分三层：正则快路径 → 本地模型推理 → 低置信度 LLM 兜底。

意图类别：`query_weather` / `query_attraction` / `query_hotel` / `plan_change` / `other`。

复现训练：

​```bash
cd backend
python -m ml.intent.data_gen   # 用 LLM 合成训练数据 → ml/intent/data/train.jsonl
python -m ml.intent.train      # 微调并保存到 models/intent_classifier/
​```

模型产物与合成数据不入库；手写测试集 `ml/intent/eval.jsonl` 用于评估
（macro-F1 ≥ 0.90）。
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document finetuned intent classifier and reproduction steps"
```

---

## Self-Review 记录

- **Spec 覆盖**：数据 pipeline（T3）、手写测试集（T2）、模型训练（T4）、基座与超参（T4）、三层推理（T5+T6）、5 类路由与字段提示（T7）、依赖与 gitignore（T1）、评估标准（T4 Step5 验收）、错误降级（T6 逻辑）、README（T8）、可扩展性目录约定（本计划 File Structure 已按 `ml/<task>/` 落位）。均有对应任务。
- **Placeholder 扫描**：训练/推理代码均为完整实现；eval.jsonl 给出 50 条种子并明确扩写目标（数据编写本质为人工，非代码占位）。
- **类型一致性**：`Intent` 5 类标签在 intent_labels / intent_classifier / 测试中一致；`predict -> (label, confidence)`、`INTENT_TO_NODE`、`QUERY_INTENT_FIELD` 跨任务签名一致；`IntentModelUnavailable` 在 T5 定义、T6 使用。
