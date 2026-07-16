"""把 grounded 教师数据（Task 5 `generate.py` 产出的 records.jsonl）导出为
LLaMA-Factory sharegpt 格式的 train.json / val.json。

human 侧文本必须与线上推理/训练输入保持一致（train/serve parity）：
  compact = app.planner.compact.compact_for_planner(record["planner_context"])
  human = app.planner.context.build_grounded_planner_messages(compact)[1].content
这与 Task 5 `generate.py::process()` 里喂给 teacher 的
`teacher.ainvoke(build_grounded_planner_messages(compact))` 走的是同一条
compact + message 构造路径（`builder.compact_for_planner` 只是
`compact_for_planner` 模块函数的实例方法包装，参见 app/planner/context.py）。

运行（在 backend/ 下）：
  python -m ml.planner.datagen.export \
    --runs 260703_smoke20 260704_batch100 260705_batch1400 \
    --val-ratio 0.05 --out-dir ml/planner/llamafactory/generated
"""
import argparse
import json
import random
from pathlib import Path

from app.planner.compact import compact_for_planner
from app.planner.context import build_grounded_planner_messages
from app.planner.prompts import PLANNER_AGENT_PROMPT

RUNS_DIR = Path("ml/planner/data/runs")
OUT_DIR = Path("ml/planner/llamafactory/generated")


def make_lf_row(record: dict) -> dict:
    """把一条 grounded record 转成 sharegpt 行。

    system 固定为 PLANNER_AGENT_PROMPT；human 是 compact 后的 PlannerContext 文本
    （与 generate.py 喂给 teacher 的输入同一路径）；gpt 是落盘的 teacher_output 原文。
    """
    compact = compact_for_planner(record["planner_context"])
    human = build_grounded_planner_messages(compact)[1].content
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": record["teacher_output"]},
        ],
        "system": PLANNER_AGENT_PROMPT,
    }


def write_llamafactory_files(records: list, val_ratio: float, out_dir: str) -> tuple:
    """按 val_ratio 切分 records 并写出 train.json / val.json，返回 (train_n, val_n)。

    切分对 record 下标做固定 seed 洗牌以保证可复现；val_n = round(len(records) * val_ratio)。
    """
    idx = list(range(len(records)))
    random.Random(42).shuffle(idx)
    val_n = round(len(records) * val_ratio)
    val_idx = set(idx[:val_n])

    train_rows = [make_lf_row(records[i]) for i in range(len(records)) if i not in val_idx]
    val_rows = [make_lf_row(records[i]) for i in range(len(records)) if i in val_idx]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "train.json").write_text(json.dumps(train_rows, ensure_ascii=False, indent=1))
    (out_path / "val.json").write_text(json.dumps(val_rows, ensure_ascii=False, indent=1))

    return len(train_rows), len(val_rows)


def _load_records(slugs: list) -> list:
    records = []
    for slug in slugs:
        path = RUNS_DIR / slug / "records.jsonl"
        kept = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                kept += 1
        print(f"{slug}: 读取 {kept} 条，累计 {len(records)} 条")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    records = _load_records(args.runs)
    train_n, val_n = write_llamafactory_files(records, args.val_ratio, args.out_dir)
    print(f"train={train_n} val={val_n} -> {args.out_dir}")


if __name__ == "__main__":
    main()
