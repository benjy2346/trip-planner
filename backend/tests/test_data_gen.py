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
