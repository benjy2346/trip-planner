import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.intent_classifier import classify_by_rules, classify_intent, IntentResult
from app.agents.intent_model import IntentModelUnavailable


# --- 规则层 ---

def test_rule_weather():
    assert classify_by_rules("天气怎么样") == "query_weather"

def test_rule_hotel():
    assert classify_by_rules("第一天住哪个酒店") == "query_hotel"

def test_rule_budget_is_hotel():
    assert classify_by_rules("总费用是多少") == "query_hotel"

def test_rule_attraction():
    assert classify_by_rules("第3天景点有哪些") == "query_attraction"

def test_rule_food_is_attraction():
    assert classify_by_rules("中午吃什么") == "query_attraction"

def test_rule_other_thanks():
    assert classify_by_rules("谢谢") == "other"

def test_rule_no_match_returns_none():
    assert classify_by_rules("帮我把第二天改得轻松一点") is None


# --- classify_intent 三层 ---

@pytest.mark.asyncio
async def test_rule_first_skips_model_and_llm():
    with patch("app.agents.intent_classifier.intent_model.predict") as mp, \
         patch("app.agents.intent_classifier.get_agent_llm") as ml:
        result = await classify_intent("第一天住哪")
    mp.assert_not_called()
    ml.assert_not_called()
    assert result == "query_hotel"


@pytest.mark.asyncio
async def test_model_high_confidence_returned():
    with patch("app.agents.intent_classifier.intent_model.predict",
               return_value=("plan_change", 0.93)), \
         patch("app.agents.intent_classifier.get_agent_llm") as ml:
        result = await classify_intent("删掉第三天的博物馆换成购物")
    ml.assert_not_called()
    assert result == "plan_change"


@pytest.mark.asyncio
async def test_model_low_confidence_falls_back_to_llm():
    # LLM 返回的意图故意不同于默认兜底值 plan_change，以证明确实走了 LLM 层
    mock_result = IntentResult(intent="query_weather", confidence=0.95)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)
    with patch("app.agents.intent_classifier.intent_model.predict",
               return_value=("query_hotel", 0.4)), \
         patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("嗯那个你看着办")
    assert result == "query_weather"
    mock_llm.with_structured_output.assert_called_once()


@pytest.mark.asyncio
async def test_model_unavailable_falls_back_to_llm():
    mock_result = IntentResult(intent="query_weather", confidence=0.9)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)
    with patch("app.agents.intent_classifier.intent_model.predict",
               side_effect=IntentModelUnavailable), \
         patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("那几天会不会冷")
    assert result == "query_weather"


@pytest.mark.asyncio
async def test_llm_exception_defaults_to_plan_change():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=Exception("boom"))
    with patch("app.agents.intent_classifier.intent_model.predict",
               side_effect=IntentModelUnavailable), \
         patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("随便改改")
    assert result == "plan_change"
    # 证明确实到达了 LLM 层（异常后才默认 plan_change），而非提前静默默认
    mock_llm.with_structured_output.assert_called_once()
