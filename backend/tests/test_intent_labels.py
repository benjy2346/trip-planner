from app.agents.intent_labels import (
    INTENT_LABELS, LABEL2ID, ID2LABEL, INTENT_TO_NODE, QUERY_INTENT_FIELD,
)


def test_five_labels_fixed_order():
    assert INTENT_LABELS == [
        "query_weather", "query_attraction", "query_hotel", "plan_change", "other",
    ]


def test_label_id_roundtrip():
    for i, label in enumerate(INTENT_LABELS):
        assert LABEL2ID[label] == i
        assert ID2LABEL[i] == label


def test_every_label_routes_to_a_node():
    for label in INTENT_LABELS:
        assert INTENT_TO_NODE[label] in {"query_handler", "modify_handler", "other_handler"}


def test_query_intents_map_to_fields():
    assert QUERY_INTENT_FIELD == {
        "query_weather": "weather",
        "query_attraction": "attraction",
        "query_hotel": "hotel",
    }
