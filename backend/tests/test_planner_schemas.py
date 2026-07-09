import pytest
from pydantic import ValidationError
from app.models.schemas import TripRequest, PartyInfo, BudgetConstraint


def _base_kwargs():
    return dict(
        city="北京", start_date="2026-08-01", end_date="2026-08-03",
        travel_days=3, transportation="公共交通", accommodation="经济型酒店",
    )


def test_trip_request_backward_compatible_defaults():
    req = TripRequest(**_base_kwargs())
    assert req.party.adults == 1
    assert req.party.total == 1
    assert req.budget_constraint is None


def test_party_total_computed():
    p = PartyInfo(adults=2, children=1, elders=1)
    assert p.total == 4
    assert p.model_dump()["total"] == 4


def test_party_rejects_zero_adults():
    with pytest.raises(ValidationError):
        PartyInfo(adults=0)


def test_budget_constraint_fields():
    b = BudgetConstraint(amount=3500, budget_level="limited", strictness="hard")
    assert b.scope == "total"
    assert b.currency == "CNY"


def test_budget_rejects_bad_level():
    with pytest.raises(ValidationError):
        BudgetConstraint(budget_level="luxury")


def test_trip_request_accepts_structured_fields():
    req = TripRequest(
        **_base_kwargs(),
        party={"adults": 2, "children": 1},
        budget_constraint={"amount": 5000, "strictness": "hard"},
    )
    assert req.party.total == 3
    assert req.budget_constraint.amount == 5000
