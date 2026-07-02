from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from app.agents.state_trimmer import trim_state, WINDOW_SIZE


def _make_state(n: int, trip_plan=None, summary=""):
    msgs = [HumanMessage(content=f"msg {i}") for i in range(n)]
    return {
        "messages": msgs,
        "trip_plan": trip_plan,
        "summary": summary,
        "trip_request": None,
        "weather_outputs": [],
        "hotel_outputs": [],
        "poi_outputs": [],
    }


def test_no_trim_when_under_window():
    state = _make_state(3)
    mock_llm = MagicMock()
    result = trim_state(state, mock_llm)
    assert len(result["messages"]) == 3
    mock_llm.invoke.assert_not_called()


def test_trim_keeps_window_size_messages():
    state = _make_state(12)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="摘要")
    result = trim_state(state, mock_llm)
    assert len(result["messages"]) == WINDOW_SIZE


def test_trip_plan_preserved_after_trim():
    sentinel = {"city": "Beijing"}
    state = _make_state(12, trip_plan=sentinel)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="摘要")
    result = trim_state(state, mock_llm)
    assert result["trip_plan"] is sentinel


def test_summary_updated_after_trim():
    state = _make_state(12, summary="旧摘要")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="新摘要")
    result = trim_state(state, mock_llm)
    assert result["summary"] == "新摘要"


def test_no_trim_exactly_at_window():
    state = _make_state(WINDOW_SIZE)
    mock_llm = MagicMock()
    result = trim_state(state, mock_llm)
    assert len(result["messages"]) == WINDOW_SIZE
    mock_llm.invoke.assert_not_called()
