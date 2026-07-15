import json
from app.models.schemas import TripRequest
from ml.planner.datagen.leakage import eval_signature, load_eval_signatures

EVAL = "ml/planner/eval/records.jsonl"


def test_eval_request_is_flagged():
    sigs = load_eval_signatures([EVAL])
    assert len(sigs) == 200
    with open(EVAL, encoding="utf-8") as f:
        first_req = TripRequest(**json.loads(f.readline())["request"])
    assert eval_signature(first_req) in sigs


def test_unrelated_request_not_flagged():
    sigs = load_eval_signatures([EVAL])
    novel = TripRequest(city="张家界", start_date="2019-01-01", end_date="2019-01-02", travel_days=2,
                        transportation="打车", accommodation="经济型酒店", preferences=["摄影"],
                        party={"adults": 1}, budget_constraint={"amount": 1234, "strictness": "soft"})
    assert eval_signature(novel) not in sigs
