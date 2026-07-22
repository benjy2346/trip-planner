import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "export_lf", Path(__file__).resolve().parent.parent / "ml" / "planner" / "export_llamafactory.py")
export_lf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_lf)


def test_to_sharegpt_structure():
    record = {"record_id": "x", "context": {"request": {"city": "北京"}},
              "teacher_output": '{"city":"北京"}'}
    row = export_lf.to_sharegpt(record)
    assert set(row.keys()) == {"conversations", "system"}
    assert row["conversations"][0]["from"] == "human"
    assert "北京" in row["conversations"][0]["value"]
    assert row["conversations"][1] == {"from": "gpt", "value": '{"city":"北京"}'}


def test_split_deterministic_and_disjoint():
    rows = [{"id": i} for i in range(100)]
    t1, v1 = export_lf.split_rows(rows, val_ratio=0.05, seed=42)
    t2, v2 = export_lf.split_rows(rows, val_ratio=0.05, seed=42)
    assert t1 == t2 and v1 == v2
    assert len(v1) == 5
    assert not {id(x) for x in t1} & {id(x) for x in v1}


def _plan_with_meals(lunch_dinner_names):
    """lunch_dinner_names: list of (lunch, dinner) per day."""
    days = []
    for i, (lunch, dinner) in enumerate(lunch_dinner_names):
        days.append({"meals": [
            {"type": "breakfast", "name": f"早点{i}"},
            {"type": "lunch", "name": lunch},
            {"type": "dinner", "name": dinner},
        ]})
    return json.dumps({"days": days}, ensure_ascii=False)


def test_meal_repeat_count():
    clean = _plan_with_meals([("面馆A", "菜馆B"), ("面馆C", "菜馆D")])
    assert export_lf.meal_repeat_count(clean) == 0
    dup = _plan_with_meals([("堂宴", "临家"), ("堂宴", "临家")])
    assert export_lf.meal_repeat_count(dup) == 2
    # 早餐重复不计（只看午晚餐）
    assert export_lf.meal_repeat_count(clean) == 0
    # 无法解析的输出按 0 处理（不误伤，交由上游硬过滤）
    assert export_lf.meal_repeat_count("{坏json") == 0
