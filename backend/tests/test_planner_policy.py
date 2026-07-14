from app.models.schemas import TripRequest
from app.planner.policy import build_preference_profile


def _req(free_text, prefs=None, party=None):
    return TripRequest(city="北京", start_date="2026-08-01", end_date="2026-08-05",
                       travel_days=5, transportation="打车", accommodation="经济型酒店",
                       preferences=prefs or ["历史文化"], free_text_input=free_text,
                       party=party or {"adults": 2, "elders": 1})


def test_elder_and_no_spicy_and_avoid_walk_parsed():
    # NOTE: policy.py's avoid_long_walk markers are
    # ["少走路", "不想太累", "行动不便", "无障碍", "轮椅"] — "少爬山" does not
    # match any of them (it only feeds NEGATIVE_CONSTRAINT_PHRASES via "爬山").
    # Use "少走路" so this test exercises the real avoid_long_walk marker set.
    p = build_preference_profile(_req("有老人同行，少走路，不吃辣"))
    assert "不吃辣" in p["diet_avoid"] or "辣" in "".join(p["diet_avoid"])
    assert p["traveler_constraints"]["avoid_long_walk"] is True
    assert p["traveler_constraints"]["needs_elder_friendly"] is True


def test_positive_preferences_carried():
    p = build_preference_profile(_req("", prefs=["美食", "博物馆"]))
    assert "美食" in p["positive_preferences"]
