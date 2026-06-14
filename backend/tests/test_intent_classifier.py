import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.intent_classifier import classify_by_rules, classify_intent


# --- 规则层测试 ---

def test_rule_query_day():
    assert classify_by_rules("第一天住哪个酒店") == "query_plan"

def test_rule_query_digit_day():
    assert classify_by_rules("第3天景点有哪些") == "query_plan"

def test_rule_query_budget():
    assert classify_by_rules("总费用是多少") == "query_plan"

def test_rule_query_weather():
    assert classify_by_rules("天气怎么样") == "query_plan"

def test_rule_other_thanks():
    assert classify_by_rules("谢谢") == "other"

def test_rule_other_greeting():
    assert classify_by_rules("你好") == "other"

def test_rule_no_match_returns_none():
    assert classify_by_rules("帮我把第二天改得轻松一点") is None

def test_rule_no_match_complex():
    assert classify_by_rules("删掉第三天的博物馆，换成购物中心") is None


# --- LLM 分类层测试 ---

@pytest.mark.asyncio
async def test_classify_intent_uses_rule_first():
    with patch("app.agents.intent_classifier.get_agent_llm") as mock:
        result = await classify_intent("第一天住哪")
    mock.assert_not_called()
    assert result == "query_plan"


@pytest.mark.asyncio
async def test_classify_intent_llm_modify():
    from app.agents.intent_classifier import IntentResult
    mock_result = IntentResult(intent="modify", confidence=0.95)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)

    with patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("帮我改一下第二天行程")
    assert result == "modify"


@pytest.mark.asyncio
async def test_classify_intent_low_confidence_fallback_to_modify():
    from app.agents.intent_classifier import IntentResult
    mock_result = IntentResult(intent="query_plan", confidence=0.5)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)

    with patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("随便改改")
    assert result == "modify"


@pytest.mark.asyncio
async def test_classify_intent_llm_exception_fallback():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

    with patch("app.agents.intent_classifier.get_agent_llm", return_value=mock_llm):
        result = await classify_intent("改一下行程")
    assert result == "modify"
