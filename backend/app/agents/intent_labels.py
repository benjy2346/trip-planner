"""意图标签与路由映射（训练与推理共用的单一来源）"""

INTENT_LABELS: list[str] = [
    "query_weather",
    "query_attraction",
    "query_hotel",
    "plan_change",
    "other",
]

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

INTENT_TO_NODE: dict[str, str] = {
    "query_weather": "query_handler",
    "query_attraction": "query_handler",
    "query_hotel": "query_handler",
    "plan_change": "modify_handler",
    "other": "other_handler",
}

QUERY_INTENT_FIELD: dict[str, str] = {
    "query_weather": "weather",
    "query_attraction": "attraction",
    "query_hotel": "hotel",
}
