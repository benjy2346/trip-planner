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
