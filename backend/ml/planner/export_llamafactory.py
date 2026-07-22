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


def meal_repeat_count(teacher_output: str) -> int:
    """午晚餐店名重复计数（soft 质量）。解析失败按 0 处理，交由上游硬过滤兜底。"""
    try:
        data = json.loads(teacher_output)
    except Exception:
        return 0
    names = [m.get("name") for d in data.get("days", []) for m in d.get("meals", [])
             if m.get("type") in ("lunch", "dinner")]
    return len(names) - len(set(names))


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
    ap.add_argument("--allow-meal-repeat", action="store_true",
                    help="默认丢弃午晚餐重复的记录（只保留 soft_pass 级质量）；加此项则不过滤")
    args = ap.parse_args()

    rows = []
    total_skipped = 0
    for slug in args.runs:
        path = RUNS_DIR / slug / "records.jsonl"
        kept = skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not args.allow_meal_repeat and meal_repeat_count(record["teacher_output"]) > 0:
                    skipped += 1
                    continue
                rows.append(to_sharegpt(record))
                kept += 1
        total_skipped += skipped
        print(f"{slug}: 保留 {kept} 条，丢弃重复 {skipped} 条，累计 {len(rows)} 条")
    if total_skipped:
        print(f"共丢弃午晚餐重复记录 {total_skipped} 条（--allow-meal-repeat 可关闭过滤）")

    train, val = split_rows(rows, args.val_ratio)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=1))
    (OUT_DIR / "val.json").write_text(json.dumps(val, ensure_ascii=False, indent=1))
    print(f"train={len(train)} val={len(val)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
