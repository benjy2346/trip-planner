from app.models.schemas import TripRequest, WeatherInfo, Hotel, Attraction, Location
from app.planner.context import (
    build_planner_context, build_planner_messages, PLANNER_SYSTEM_PROMPT,
)


def _req():
    return TripRequest(
        city="杭州", start_date="2026-08-01", end_date="2026-08-03",
        travel_days=3, transportation="打车", accommodation="经济型酒店",
        preferences=["美食"], free_text_input="预算3500左右",
        party={"adults": 2}, budget_constraint={"amount": 3500, "strictness": "hard"},
    )


def _snapshot_inputs():
    weather = [WeatherInfo(date="2026-08-01", day_weather="晴", night_weather="多云",
                           day_temp=30, night_temp=22)]
    hotels = [Hotel(name="如家杭州店", address="上城区", estimated_cost=250, type="经济型")]
    pois = [Attraction(name="西湖", address="西湖区", location=Location(longitude=120.1, latitude=30.2),
                       visit_duration=120, description="湖", ticket_price=0)]
    return weather, hotels, pois


def test_context_has_all_sections():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    assert set(ctx.keys()) == {
        "request", "party", "budget_constraint", "lodging_policy",
        "pricing_policy", "tool_snapshot", "planner_constraints",
    }


def test_dates_and_lodging():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    assert ctx["planner_constraints"]["dates"] == ["2026-08-01", "2026-08-02", "2026-08-03"]
    assert ctx["lodging_policy"]["nights"] == 2
    assert ctx["lodging_policy"]["hotel_on_last_day"] is False


def test_party_and_budget_compiled():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    assert ctx["party"]["total"] == 2
    assert ctx["budget_constraint"]["amount"] == 3500
    assert ctx["budget_constraint"]["strictness"] == "hard"


def test_default_budget_when_absent():
    req = _req()
    req.budget_constraint = None
    ctx = build_planner_context(req, *_snapshot_inputs())
    assert ctx["budget_constraint"]["amount"] is None
    assert ctx["budget_constraint"]["strictness"] == "soft"


def test_snapshot_counts():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    counts = ctx["tool_snapshot"]["candidate_counts"]
    assert counts == {"weather": 1, "hotels": 1, "attractions": 1}


def test_prompt_bans_fake_distance_and_placeholders():
    assert "距离景点2公里" not in PLANNER_SYSTEM_PROMPT
    assert "占位" in PLANNER_SYSTEM_PROMPT
    assert '"distance": ""' in PLANNER_SYSTEM_PROMPT


def test_messages_carry_context_json():
    ctx = build_planner_context(_req(), *_snapshot_inputs())
    msgs = build_planner_messages(ctx)
    assert len(msgs) == 2
    assert msgs[0].content == PLANNER_SYSTEM_PROMPT
    assert "西湖" in msgs[1].content and "如家杭州店" in msgs[1].content
