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
