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
