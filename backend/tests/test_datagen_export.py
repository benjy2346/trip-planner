from app.planner.prompts import PLANNER_AGENT_PROMPT
from ml.planner.datagen.export import make_lf_row, write_llamafactory_files


def _rec(city="杭州"):
    return {"record_id": "r1",
            "planner_context": {"request": {"city": city}, "tool_snapshot": {"food_pois": [{"name": "外婆家"}]}},
            "teacher_output": '{"city":"' + city + '","days":[]}'}


def test_make_lf_row_has_prompt_and_output():
    row = make_lf_row(_rec())
    assert row["system"] == PLANNER_AGENT_PROMPT
    human = row["conversations"][0]["value"]
    gpt = row["conversations"][1]["value"]
    assert "外婆家" in human
    assert '"city":"杭州"' in gpt


def test_split_ratio(tmp_path):
    recs = [_rec(city=f"城{i}") for i in range(20)]
    train_n, val_n = write_llamafactory_files(recs, val_ratio=0.1, out_dir=str(tmp_path))
    assert train_n == 18 and val_n == 2
    assert (tmp_path / "train.json").exists() and (tmp_path / "val.json").exists()
