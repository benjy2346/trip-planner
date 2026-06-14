from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from openai import APITimeoutError, RateLimitError, APIConnectionError, InternalServerError, BadRequestError
from app.config import get_settings

_FALLBACK_ERRORS = (APITimeoutError, RateLimitError, APIConnectionError, TimeoutError, InternalServerError, BadRequestError)
_AGENTS_CONFIG_PATH = Path(__file__).parent.parent.parent / "agents_config.yaml"

_llm_chain: Runnable | None = None
_primary_llm: ChatOpenAI | None = None
_agent_llm_cache: dict[str, ChatOpenAI] = {}


def _make_llm(base_url: str, api_key: str, model: str, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key or "placeholder",
        model=model,
        temperature=temperature,
        timeout=get_settings().llm_timeout,
    )


def _load_agents_config() -> dict:
    import yaml
    with open(_AGENTS_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_agent_llm(agent_name: str) -> ChatOpenAI:
    if agent_name in _agent_llm_cache:
        return _agent_llm_cache[agent_name]
    config = _load_agents_config()["agents"][agent_name]
    s = get_settings()
    base_url = getattr(s, f"{config['provider']}_base_url")
    api_key = getattr(s, f"{config['provider']}_api_key")
    llm = _make_llm(base_url, api_key, config["model"], config.get("temperature", 0.7))
    _agent_llm_cache[agent_name] = llm
    return llm


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
    global _llm_chain, _primary_llm
    if _primary_llm is None:
        _llm_chain, _primary_llm = _build_chain()
    return _primary_llm


def get_structured_chain(schema: type) -> Runnable:
    s = get_settings()
    primary = _make_llm(s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model)
    gemini = _make_llm(s.gemini_base_url, s.gemini_api_key, s.gemini_model)
    openai_llm = _make_llm(s.openai_base_url, s.openai_api_key, s.openai_model)
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
