from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from openai import APITimeoutError, RateLimitError, APIConnectionError
from app.config import get_settings

_FALLBACK_ERRORS = (APITimeoutError, RateLimitError, APIConnectionError, TimeoutError)

_llm_chain: Runnable | None = None
_primary_llm: ChatOpenAI | None = None


def _build_chain() -> tuple[Runnable, ChatOpenAI]:
    s = get_settings()
    primary = ChatOpenAI(
        base_url=s.deepseek_base_url,
        api_key=s.deepseek_api_key or "placeholder",
        model=s.deepseek_model,
        timeout=8,
    )
    gemini = ChatOpenAI(
        base_url=s.gemini_base_url,
        api_key=s.gemini_api_key,
        model=s.gemini_model,
        timeout=8,
    )
    openai = ChatOpenAI(
        base_url=s.openai_base_url,
        api_key=s.openai_api_key or "placeholder",
        model=s.openai_model,
        timeout=8,
    )
    chain = primary.with_fallbacks(
        [gemini, openai],
        exceptions_to_handle=_FALLBACK_ERRORS,
    )
    return chain, primary


def get_llm_chain() -> Runnable:
    global _llm_chain, _primary_llm
    if _llm_chain is None:
        _llm_chain, _primary_llm = _build_chain()
    return _llm_chain


def get_primary_llm() -> ChatOpenAI:
    """返回主供应商 LLM，用于非关键调用（如 state trimmer）。"""
    global _llm_chain, _primary_llm
    if _primary_llm is None:
        _llm_chain, _primary_llm = _build_chain()
    return _primary_llm


def call_with_fallback(messages: list[BaseMessage]):
    return get_llm_chain().invoke(messages)


async def acall_with_fallback(messages: list[BaseMessage]):
    return await get_llm_chain().ainvoke(messages)
