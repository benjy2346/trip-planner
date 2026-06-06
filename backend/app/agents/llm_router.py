from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from openai import APITimeoutError, RateLimitError, APIConnectionError
from app.config import get_settings


def _build_providers() -> list[ChatOpenAI]:
    s = get_settings()
    return [
        ChatOpenAI(
            base_url=s.deepseek_base_url,
            api_key=s.deepseek_api_key or "placeholder",
            model=s.deepseek_model,
            timeout=8,
        ),
        ChatOpenAI(
            base_url=s.gemini_base_url,
            api_key=s.gemini_api_key,
            model=s.gemini_model,
            timeout=8,
        ),
        ChatOpenAI(
            base_url=s.openai_base_url,
            api_key=s.openai_api_key or "placeholder",
            model=s.openai_model,
            timeout=8,
        ),
    ]


_FALLBACK_ERRORS = (APITimeoutError, RateLimitError, APIConnectionError, TimeoutError)


def call_with_fallback(messages: list[BaseMessage]):
    last_err = None
    for llm in _build_providers():
        try:
            return llm.invoke(messages)
        except _FALLBACK_ERRORS as e:
            last_err = e
    raise RuntimeError(f"All LLM providers failed. Last: {last_err}")


async def acall_with_fallback(messages: list[BaseMessage]):
    last_err = None
    for llm in _build_providers():
        try:
            return await llm.ainvoke(messages)
        except _FALLBACK_ERRORS as e:
            last_err = e
    raise RuntimeError(f"All LLM providers failed. Last: {last_err}")
