import json
from app.models.schemas import TripRequest
from ml.planner.datagen.generate import assemble_record, is_clean

_PLAN = ('{"city":"杭州","start_date":"2020-04-01","end_date":"2020-04-01",'
         '"days":[],"weather_info":[],"overall_suggestions":"ok",'
         '"budget":{"total_attractions":0,"total_hotels":0,"total_meals":0,"total_transportation":0,"total":0}}')


def _ctx():
    return {"request": {"city": "杭州", "start_date": "2020-04-01", "end_date": "2020-04-01"},
            "party": {"total": 2}, "preference_profile": {"diet_avoid": []},
            "planner_constraints": {"expected_dates": ["2020-04-01"]},
            "tool_snapshot": {"trip_weather": [], "classic_pois": [], "preference_pois": [],
                              "scenic_pois": [], "experience_pois": [], "hotel_pois": [], "food_pois": []}}


def test_is_clean_flags_day_count_mismatch():
    # context 期望 1 天，plan 给 0 天 → 有违规 → 不干净
    ok, violations = is_clean(_PLAN, _ctx())
    assert ok is False and violations


def test_assemble_record_shape():
    item = {"city": "杭州", "control_spec": {"budget_level": "standard"}}
    rec = assemble_record(item | {"record_id": "sft_test_0001"}, _ctx(), _PLAN)
    assert rec["record_id"] == "sft_test_0001"
    assert rec["control_spec"]["budget_level"] == "standard"
    assert rec["planner_context"] == _ctx()
    assert json.loads(rec["teacher_output"])["city"] == "杭州"
