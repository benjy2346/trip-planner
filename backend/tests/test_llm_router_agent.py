import pytest
from unittest.mock import patch
from langchain_openai import ChatOpenAI


def test_get_agent_llm_returns_chat_openai():
    mock_config = {
        "agents": {
            "intent_classifier": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "temperature": 0.0,
            }
        }
    }
    with patch("app.agents.llm_router._load_agents_config", return_value=mock_config):
        from app.agents.llm_router import get_agent_llm
        llm = get_agent_llm("intent_classifier")
    assert isinstance(llm, ChatOpenAI)


def test_get_agent_llm_cached():
    mock_config = {
        "agents": {
            "intent_classifier": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "temperature": 0.0,
            }
        }
    }
    with patch("app.agents.llm_router._load_agents_config", return_value=mock_config):
        from app.agents.llm_router import get_agent_llm, _agent_llm_cache
        _agent_llm_cache.clear()
        llm1 = get_agent_llm("intent_classifier")
        llm2 = get_agent_llm("intent_classifier")
    assert llm1 is llm2


@pytest.mark.asyncio
async def test_acall_agent_with_fallback_uses_agent_llm():
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.agents.llm_router import acall_agent_with_fallback

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value="agent-reply")
    with patch("app.agents.llm_router.get_agent_llm", return_value=mock_llm), \
         patch("app.agents.llm_router.acall_with_fallback", AsyncMock()) as mock_global:
        result = await acall_agent_with_fallback("assembler", ["msg"])
    assert result == "agent-reply"
    mock_global.assert_not_called()


@pytest.mark.asyncio
async def test_acall_agent_with_fallback_degrades_to_global_chain():
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.agents.llm_router import acall_agent_with_fallback

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError("down"))
    with patch("app.agents.llm_router.get_agent_llm", return_value=mock_llm), \
         patch("app.agents.llm_router.acall_with_fallback",
               AsyncMock(return_value="global-reply")) as mock_global:
        result = await acall_agent_with_fallback("assembler", ["msg"])
    assert result == "global-reply"
    mock_global.assert_called_once()

