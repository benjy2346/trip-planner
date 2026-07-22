from datetime import date

from app.models.schemas import TripRequest
from app.planner.pois import infer_food_constraints
from ml.planner.datagen.requests import (
    AVOID_LONG_WALK_MARKERS,
    DIET_WEIGHTS,
    generate_controlled_request,
    iter_requests,
    to_trip_request,
)


def test_seed_reproducible():
    a = iter_requests(10, seed=9200, date_mode="mixed")
    b = iter_requests(10, seed=9200, date_mode="mixed")
    assert [x["city"] for x in a] == [x["city"] for x in b]


def test_control_spec_present():
    items = iter_requests(5, seed=9200, date_mode="mixed")
    for it in items:
        cs = it["control_spec"]
        for k in ("city_tier", "companion_type", "budget_level", "budget_strictness"):
            assert k in cs


def test_mixed_mode_includes_past_dates():
    items = iter_requests(40, seed=9200, date_mode="mixed")
    today = date.today()
    assert any(date.fromisoformat(it["end_date"]) < today for it in items)


def test_to_trip_request_valid():
    item = iter_requests(1, seed=9200, date_mode="mixed")[0]
    req = to_trip_request(item)
    assert isinstance(req, TripRequest)
    assert req.travel_days >= 1


def test_diet_avoid_round_trips_for_every_diet_label():
    """control_spec.diet_avoid 必须和 app.planner.pois.infer_food_constraints 从真实
    TripRequest（free_text_input + preferences）里实际抽出的 avoid 完全一致，否则声明的
    忌口在下游 preference_profile 里就是死的（Task 5 消费的是 infer_food_constraints
    的输出，不是 control_spec 本身）。

    对 DIET_WEIGHTS 里的每一个 label 都跑到至少一条样本，逐条断言相等，而不是只测
    某一个 seed 恰好抽到的 label。
    """
    target_labels = {label for label, _ in DIET_WEIGHTS}
    covered: dict[str, dict] = {}
    index = 0
    # seed=9200 的 DIET_WEIGHTS 分布下，几千条以内必定能覆盖到全部 5 个 label
    # （最小权重的 label 概率约 4%）。给足上限避免死循环，同时不依赖运气。
    while len(covered) < len(target_labels) and index < 20000:
        item = generate_controlled_request(index, seed=9200, date_mode="mixed")
        label = item["control_spec"]["diet"]
        covered.setdefault(label, item)
        index += 1

    assert covered.keys() == target_labels, f"未覆盖到全部 diet label：缺 {target_labels - covered.keys()}"

    for label, item in covered.items():
        cs = item["control_spec"]
        derived = infer_food_constraints(to_trip_request(item))
        assert cs["diet_avoid"] == derived["avoid"], (
            f"diet label={label!r} 的 control_spec.diet_avoid={cs['diet_avoid']!r} 和真实"
            f" infer_food_constraints 算出的 avoid={derived['avoid']!r} 不一致"
        )
        assert cs["diet_positive"] == ([] if derived["diet"] == "无" else [derived["diet"]])


def test_avoid_long_walk_round_trips_when_declared():
    """control_spec.traveler_constraints.avoid_long_walk=True 时，free_text_input 里必须
    真的含有 app.planner.policy 实际识别的移动性关键词之一，否则这条约束在真实
    preference_profile 里永远是 False。"""
    items = iter_requests(200, seed=9200, date_mode="mixed")
    positive = [it for it in items if it["control_spec"]["traveler_constraints"]["avoid_long_walk"]]
    assert positive, "样本里应该至少有一条声明了 avoid_long_walk"
    for it in positive:
        free_text = it["free_text_input"]
        assert any(marker in free_text for marker in AVOID_LONG_WALK_MARKERS), (
            f"声明了 avoid_long_walk=True 但 free_text_input 里没有任何真实识别词：{free_text!r}"
        )
