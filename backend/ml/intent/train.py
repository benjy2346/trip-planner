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
        # padding 到固定长度，使默认 collator 得到等长张量（否则变长文本会在第一个 batch 崩溃）
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

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
