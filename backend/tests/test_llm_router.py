import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.asyncio
async def test_acall_returns_chain_result():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    with patch("app.agents.llm_router.get_llm_chain", return_value=mock_chain):
        from app.agents.llm_router import acall_with_fallback
        result = await acall_with_fallback([HumanMessage(content="hi")])

    assert result.content == "ok"
    mock_chain.ainvoke.assert_called_once_with([HumanMessage(content="hi")])


@pytest.mark.asyncio
async def test_acall_propagates_exception():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("All providers failed"))

    with patch("app.agents.llm_router.get_llm_chain", return_value=mock_chain):
        from app.agents.llm_router import acall_with_fallback
        with pytest.raises(RuntimeError, match="All providers failed"):
            await acall_with_fallback([HumanMessage(content="hi")])


def test_get_llm_chain_returns_singleton():
    with patch("app.agents.llm_router._llm_chain", None):
        with patch("app.agents.llm_router._build_chain") as mock_build:
            mock_chain = MagicMock()
            mock_primary = MagicMock()
            mock_build.return_value = (mock_chain, mock_primary)

            from app.agents.llm_router import get_llm_chain
            chain1 = get_llm_chain()
            chain2 = get_llm_chain()

    mock_build.assert_called_once()  # 只构建一次
    assert chain1 is chain2
