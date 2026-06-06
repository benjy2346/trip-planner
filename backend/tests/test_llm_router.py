import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from openai import APITimeoutError, RateLimitError


@pytest.mark.asyncio
async def test_acall_uses_first_provider_when_healthy():
    fast = MagicMock()
    fast.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
    slow = MagicMock()

    with patch("app.agents.llm_router._build_providers", return_value=[fast, slow]):
        from app.agents.llm_router import acall_with_fallback
        result = await acall_with_fallback([HumanMessage(content="hi")])

    assert result.content == "ok"
    slow.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_acall_falls_back_on_timeout():
    failing = MagicMock()
    failing.ainvoke = AsyncMock(side_effect=TimeoutError("timeout"))
    working = MagicMock()
    working.ainvoke = AsyncMock(return_value=AIMessage(content="fallback"))

    with patch("app.agents.llm_router._build_providers", return_value=[failing, working]):
        from app.agents.llm_router import acall_with_fallback
        result = await acall_with_fallback([HumanMessage(content="hi")])

    assert result.content == "fallback"


@pytest.mark.asyncio
async def test_acall_falls_back_on_rate_limit():
    failing = MagicMock()
    failing.ainvoke = AsyncMock(side_effect=RateLimitError(
        message="rate limit", response=MagicMock(status_code=429, headers={}), body={}
    ))
    working = MagicMock()
    working.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    with patch("app.agents.llm_router._build_providers", return_value=[failing, working]):
        from app.agents.llm_router import acall_with_fallback
        result = await acall_with_fallback([HumanMessage(content="hi")])

    assert result.content == "ok"


@pytest.mark.asyncio
async def test_acall_raises_when_all_fail():
    failing = MagicMock()
    failing.ainvoke = AsyncMock(side_effect=TimeoutError("timeout"))

    with patch("app.agents.llm_router._build_providers", return_value=[failing, failing]):
        from app.agents.llm_router import acall_with_fallback
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await acall_with_fallback([HumanMessage(content="hi")])
