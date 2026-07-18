"""generate.py 单测：验证 train/serve parity（用 compact_planner_context + PLANNER_AGENT_PROMPT）
和 generations.jsonl 落盘格式。不打真实网络。"""
import json

from app.planner.prompts import PLANNER_AGENT_PROMPT
from ml.planner.eval.generate import messages_for, write_generations


def _record():
    return {
        "record_id": "r1",
        "compact_planner_context": {"version": "planner-1", "request": {"city": "杭州"}},
    }


def test_messages_use_compact_context_and_planner_prompt():
    msgs = messages_for(_record())
    assert len(msgs) == 2
    assert msgs[0].content == PLANNER_AGENT_PROMPT           # system parity
    assert "PlannerContext:" in msgs[1].content
    assert "杭州" in msgs[1].content                          # compact context 被塞进 human


def test_write_generations_format(tmp_path):
    results = [(_record(), '{"city": "杭州"}')]
    n = write_generations(results, str(tmp_path))
    assert n == 1
    lines = (tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    assert row == {"record_id": "r1", "output": '{"city": "杭州"}'}


def test_write_generations_skips_exceptions(tmp_path):
    results = [(_record(), '{"ok": 1}'), RuntimeError("boom")]
    n = write_generations(results, str(tmp_path))
    assert n == 1  # 异常项被跳过，不落盘
