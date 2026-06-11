from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from openai import APITimeoutError, RateLimitError, APIConnectionError, InternalServerError, BadRequestError
from app.config import get_settings

_FALLBACK_ERRORS = (APITimeoutError, RateLimitError, APIConnectionError, TimeoutError, InternalServerError, BadRequestError)

_llm_chain: Runnable | None = None
_primary_llm: ChatOpenAI | None = None


def _make_llm(base_url: str, api_key: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key or "placeholder",
        model=model,
        timeout=get_settings().llm_timeout,
    )


def _build_chain() -> tuple[Runnable, ChatOpenAI]:
    s = get_settings()
    primary = _make_llm(s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model)
    return primary, primary


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


def get_structured_chain(schema: type) -> Runnable:
    """每个供应商独立绑定 schema，用 function_calling 兼容 DeepSeek 等非 OpenAI 供应商。"""
    primary, gemini, openai_llm = _get_providers()
    return primary.with_structured_output(schema, method="function_calling").with_fallbacks(
        [
            gemini.with_structured_output(schema, method="function_calling"),
            openai_llm.with_structured_output(schema, method="function_calling"),
        ],
        exceptions_to_handle=_FALLBACK_ERRORS,
    )


def call_with_fallback(messages: list[BaseMessage]):
    return get_llm_chain().invoke(messages)


async def acall_with_fallback(messages: list[BaseMessage]):
    return await get_llm_chain().ainvoke(messages)
