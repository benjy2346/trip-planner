import json
import importlib.util
from pathlib import Path

_here = Path(__file__).resolve().parent.parent / "ml" / "planner"
_spec = importlib.util.spec_from_file_location("planner_data_gen", _here / "data_gen.py")
data_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_gen)

_rspec = importlib.util.spec_from_file_location("requestgen", _here / "requestgen.py")
requestgen = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(requestgen)


def test_load_eval_signatures(tmp_path):
    reqs = requestgen.iter_controlled_requests(3, "standard", seed=5)
    p = tmp_path / "records.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, r in enumerate(reqs):
            f.write(json.dumps({"record_id": f"s_{i}", "difficulty": "standard",
                                "request": r.model_dump(), "context": {}},
                               ensure_ascii=False) + "\n")
    sigs = data_gen.load_eval_signatures([str(p)])
    assert len(sigs) == 3
    assert requestgen.eval_signature(reqs[0]) in sigs


def test_extract_usage_from_metadata():
    class R:
        response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    u = data_gen.extract_usage(R())
    assert u == {"prompt_tokens": 100, "completion_tokens": 50}

    class Empty:
        response_metadata = {}
    assert data_gen.extract_usage(Empty()) == {"prompt_tokens": 0, "completion_tokens": 0}
