import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "requestgen", Path(__file__).resolve().parent.parent / "ml" / "planner" / "requestgen.py")
requestgen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(requestgen)


def test_deterministic_with_same_seed():
    a = requestgen.iter_controlled_requests(10, "standard", seed=7)
    b = requestgen.iter_controlled_requests(10, "standard", seed=7)
    assert [r.model_dump(exclude={"user_id"}) for r in a] == \
           [r.model_dump(exclude={"user_id"}) for r in b]


def test_standard_profile():
    for r in requestgen.iter_controlled_requests(20, "standard", seed=1):
        assert 2 <= r.travel_days <= 4
        assert r.party.total <= 2


def test_hard_profile():
    for r in requestgen.iter_controlled_requests(20, "hard", seed=1):
        assert 4 <= r.travel_days <= 6
        assert r.party.total >= 3
        assert r.budget_constraint is not None
        assert r.budget_constraint.strictness == "hard"


def test_signature_stable_and_distinct():
    reqs = requestgen.iter_controlled_requests(30, "standard", seed=2)
    sigs = [requestgen.eval_signature(r) for r in reqs]
    assert sigs[0] == requestgen.eval_signature(reqs[0])
    assert len(set(sigs)) > 25  # 基本不重复
