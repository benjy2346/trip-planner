# Intent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `/api/chat/modify` 前加入意图路由层（规则 → LLM 分类），并把对话修改流程迁移进新的 `chat_graph` LangGraph 节点，消除对查询和闲聊消息的无效 LLM 调用。

**Architecture:** 新建 `chat_graph`（`classify_intent` → `query_handler` / `modify_handler` / `other_handler`），与现有 `supervisor_graph` 共享 checkpointer 和 `thread_id`，通过 Redis 中的同一份 state 传递数据。分类使用两层：正则规则层（0 cost）→ DeepSeek LLM 结构化输出（低 cost，仅在规则未命中时触发）。

**Tech Stack:** LangGraph 1.2.4, `langgraph.types.Command`, `langchain_openai.ChatOpenAI`, `pydantic`, `pyyaml`, `pytest`, `fakeredis`

---

## File Map

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `backend/agents_config.yaml` | 每个 agent 的 provider/model/temperature |
| 修改 | `backend/requirements.txt` | 添加 `pyyaml` |
| 修改 | `backend/app/agents/llm_router.py` | 新增 `get_agent_llm()`，更新 `_make_llm` 支持 temperature |
| 新建 | `backend/app/agents/intent_classifier.py` | 规则层 + LLM 分类器 |
| 新建 | `backend/app/agents/chat_graph.py` | 4 个节点 + `create_chat_graph()` |
| 修改 | `backend/app/agents/__init__.py` | 注册 `init_chat_graph` / `get_chat_graph` |
| 修改 | `backend/app/api/main.py` | startup 初始化 `chat_graph` |
| 修改 | `backend/app/api/routes/chat.py` | `modify_trip` 改为 `chat_graph.ainvoke`，精简到 ~20 行 |
| 新建 | `backend/tests/test_intent_classifier.py` | 规则层和 LLM 分类器单测 |
| 新建 | `backend/tests/test_chat_graph_nodes.py` | 节点单测 |

---

## Task 1: `agents_config.yaml` + `get_agent_llm()`

**Files:**
- Create: `backend/agents_config.yaml`
- Modify: `backend/requirements.txt`
- Modify: `backend/app/agents/llm_router.py`
- Create: `backend/tests/test_llm_router_agent.py`

- [ ] **Step 1: 添加 pyyaml 依赖**

在 `backend/requirements.txt` 的 `# 工具` 区块加一行：
```
pyyaml>=6.0.0
```

安装：
```bash
pip install pyyaml
```

- [ ] **Step 2: 创建 `backend/agents_config.yaml`**

```yaml
agents:
  intent_classifier:
    provider: deepseek
    model: deepseek-chat
    temperature: 0.0

  modify_handler:
    provider: deepseek
    model: deepseek-chat
    temperature: 0.7

  assembler:
    provider: deepseek
    model: deepseek-chat
    temperature: 0.7

  state_trimmer:
    provider: deepseek
    model: deepseek-chat
    temperature: 0.0
```

- [ ] **Step 3: 写失败测试**

新建 `backend/tests/test_llm_router_agent.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认失败**

```bash
cd backend
pytest tests/test_llm_router_agent.py -v
```

Expected: `ImportError` 或 `AttributeError: get_agent_llm`

- [ ] **Step 5: 实现 `get_agent_llm`**

修改 `backend/app/agents/llm_router.py`，完整替换为：

```python
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
```

- [ ] **Step 6: 运行测试确认通过**

```bash
pytest tests/test_llm_router_agent.py -v
```

Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add backend/agents_config.yaml backend/requirements.txt backend/app/agents/llm_router.py backend/tests/test_llm_router_agent.py
git commit -m "feat: add agents_config.yaml and get_agent_llm() with per-agent model config"
```

---

## Task 2: `intent_classifier.py`

**Files:**
- Create: `backend/app/agents/intent_classifier.py`
- Create: `backend/tests/test_intent_classifier.py`

- [ ] **Step 1: 写失败测试**

新建 `backend/tests/test_intent_classifier.py`：

```python
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
    # 规则命中时不调 LLM
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_intent_classifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.agents.intent_classifier'`

- [ ] **Step 3: 实现 `intent_classifier.py`**

新建 `backend/app/agents/intent_classifier.py`：

```python
import re
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

Intent = Literal["query_plan", "modify", "other"]

QUERY_RULES: list[str] = [
    r"第[一二三四五六七八九十\d]+天",
    r"(住哪|酒店|住宿)",
    r"(景点|去哪|参观|游览)",
    r"(天气|温度|气温)",
    r"(预算|费用|花多少|多少钱)",
    r"(餐|吃什么|午餐|晚餐|早餐)",
]

OTHER_RULES: list[str] = [
    r"^(谢谢|感谢|好的|可以|没问题|好|嗯|收到)[！!。]*$",
    r"^(你好|您好|hi|hello)[！!。]*$",
]

_CLASSIFIER_PROMPT = (
    "你是行程助手的意图分类器。根据用户消息，判断意图：\n"
    "- query_plan：查询当前行程信息（天气、景点、酒店、餐饮、预算等）\n"
    "- modify：修改、调整、新增或删除行程内容\n"
    "- other：闲聊、问候、感谢等与行程无关的内容\n"
    "返回结构化 JSON，包含 intent 和 confidence（0.0-1.0）。"
)


class IntentResult(BaseModel):
    intent: Intent
    confidence: float


def classify_by_rules(message: str) -> Intent | None:
    for pattern in QUERY_RULES:
        if re.search(pattern, message):
            return "query_plan"
    for pattern in OTHER_RULES:
        if re.search(pattern, message, re.IGNORECASE):
            return "other"
    return None


async def classify_intent(message: str) -> Intent:
    rule_result = classify_by_rules(message)
    if rule_result is not None:
        return rule_result
    try:
        from app.agents.llm_router import get_agent_llm
        llm = get_agent_llm("intent_classifier")
        structured = llm.with_structured_output(IntentResult, method="function_calling")
        result: IntentResult = await structured.ainvoke([
            SystemMessage(content=_CLASSIFIER_PROMPT),
            HumanMessage(content=message),
        ])
        if result.confidence < 0.7:
            return "modify"
        return result.intent
    except Exception:
        return "modify"
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_intent_classifier.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/intent_classifier.py backend/tests/test_intent_classifier.py
git commit -m "feat: add intent_classifier with rule layer and LLM fallback"
```

---

## Task 3: `chat_graph.py` — query_handler + other_handler

**Files:**
- Create: `backend/app/agents/chat_graph.py`
- Create: `backend/tests/test_chat_graph_nodes.py`

- [ ] **Step 1: 写失败测试**

新建 `backend/tests/test_chat_graph_nodes.py`：

```python
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.state import SupervisorState
from app.models.schemas import (
    TripPlan, DayPlan, Hotel, Attraction, Meal, WeatherInfo, Budget, Location
)


def _make_state(user_message: str) -> SupervisorState:
    hotel = Hotel(
        name="北京假日酒店", address="长安街1号",
        location=Location(longitude=116.4, latitude=39.9),
        price_range="300-500元", rating="4.5", distance="1km",
        type="经济型", estimated_cost=400,
    )
    attraction = Attraction(
        name="故宫", address="北京市东城区",
        location=Location(longitude=116.397, latitude=39.916),
        visit_duration=180, description="明清皇宫，世界文化遗产",
        category="历史文化", ticket_price=60,
    )
    meal = Meal(type="lunch", name="全聚德", description="北京烤鸭", estimated_cost=80)
    day = DayPlan(
        date="2025-06-01", day_index=0, description="游览故宫",
        transportation="地铁", accommodation="经济型",
        hotel=hotel, attractions=[attraction], meals=[meal],
    )
    weather = WeatherInfo(
        date="2025-06-01", day_weather="晴", night_weather="多云",
        day_temp=28, night_temp=18, wind_direction="南风", wind_power="1-3级",
    )
    budget = Budget(
        total_attractions=60, total_hotels=400,
        total_meals=80, total_transportation=50, total=590,
    )
    plan = TripPlan(
        city="北京", start_date="2025-06-01", end_date="2025-06-01",
        days=[day], weather_info=[weather], overall_suggestions="带好遮阳帽",
        budget=budget,
    )
    from app.models.schemas import TripRequest
    return SupervisorState(
        trip_request=TripRequest(
            user_id="u1", city="北京", start_date="2025-06-01",
            end_date="2025-06-01", travel_days=1,
            transportation="地铁", accommodation="经济型",
        ),
        messages=[HumanMessage(content=user_message)],
        trip_plan=plan,
        summary="",
        weather_outputs=[],
        hotel_outputs=[],
        poi_outputs=[],
    )


@pytest.mark.asyncio
async def test_query_handler_hotel():
    from app.agents.chat_graph import query_handler_node
    state = _make_state("第一天住哪个酒店")
    result = await query_handler_node(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "北京假日酒店" in result["messages"][0].content


@pytest.mark.asyncio
async def test_query_handler_budget():
    from app.agents.chat_graph import query_handler_node
    state = _make_state("总费用是多少")
    result = await query_handler_node(state)
    assert "590" in result["messages"][0].content


@pytest.mark.asyncio
async def test_query_handler_weather():
    from app.agents.chat_graph import query_handler_node
    state = _make_state("天气怎么样")
    result = await query_handler_node(state)
    assert "晴" in result["messages"][0].content


@pytest.mark.asyncio
async def test_query_handler_no_plan():
    from app.agents.chat_graph import query_handler_node
    from app.models.schemas import TripRequest
    state = SupervisorState(
        trip_request=TripRequest(
            user_id="u1", city="北京", start_date="2025-06-01",
            end_date="2025-06-01", travel_days=1,
            transportation="地铁", accommodation="经济型",
        ),
        messages=[HumanMessage(content="第一天住哪")],
        trip_plan=None,
        summary="", weather_outputs=[], hotel_outputs=[], poi_outputs=[],
    )
    result = await query_handler_node(state)
    assert "还没有生成行程" in result["messages"][0].content


@pytest.mark.asyncio
async def test_other_handler_returns_canned():
    from app.agents.chat_graph import other_handler_node
    state = _make_state("谢谢")
    result = await other_handler_node(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "行程助手" in result["messages"][0].content
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_chat_graph_nodes.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.agents.chat_graph'`

- [ ] **Step 3: 实现 query_handler + other_handler**

新建 `backend/app/agents/chat_graph.py`：

```python
import json
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from app.agents.state import SupervisorState
from app.agents.intent_classifier import classify_intent
from app.models.schemas import TripPlan

_DAY_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_INTENT_TO_NODE = {
    "query_plan": "query_handler",
    "modify": "modify_handler",
    "other": "other_handler",
}


def _parse_day(message: str) -> int | None:
    m = re.search(r"第(\d+)天", message)
    if m:
        return int(m.group(1))
    m = re.search(r"第([一二三四五六七八九十])天", message)
    if m:
        return _DAY_MAP.get(m.group(1))
    return None


def _build_query_reply(message: str, state: SupervisorState) -> str:
    plan: TripPlan | None = state.get("trip_plan")
    if not plan:
        return "还没有生成行程，请先规划行程。"

    if re.search(r"(天气|温度|气温)", message):
        if not plan.weather_info:
            return "暂无天气信息。"
        lines = [
            f"{w.date}：{w.day_weather}，白天 {w.day_temp}°C / 夜间 {w.night_temp}°C"
            for w in plan.weather_info
        ]
        return "天气预报：\n" + "\n".join(lines)

    if re.search(r"(预算|费用|花多少|多少钱)", message):
        b = plan.budget
        if not b:
            return "暂无预算信息。"
        return (
            f"总预算：{b.total} 元\n"
            f"  景点门票：{b.total_attractions} 元\n"
            f"  住宿：{b.total_hotels} 元\n"
            f"  餐饮：{b.total_meals} 元\n"
            f"  交通：{b.total_transportation} 元"
        )

    day = _parse_day(message)
    if day is not None:
        idx = day - 1
        if idx < 0 or idx >= len(plan.days):
            return f"行程只有 {len(plan.days)} 天，没有第 {day} 天。"
        d = plan.days[idx]

        if re.search(r"(住哪|酒店|住宿)", message):
            h = d.hotel
            if not h:
                return "暂无酒店信息。"
            return f"第{day}天住宿：{h.name}（{h.address}），约 {h.estimated_cost} 元/晚。"

        if re.search(r"(餐|吃什么|午餐|晚餐|早餐)", message):
            if not d.meals:
                return "暂无餐饮信息。"
            lines = [f"  {m.type}：{m.name}，约 {m.estimated_cost} 元" for m in d.meals]
            return f"第{day}天餐饮：\n" + "\n".join(lines)

        if re.search(r"(景点|去哪|参观|游览)", message):
            if not d.attractions:
                return "暂无景点信息。"
            lines = [f"  {a.name}（建议 {a.visit_duration} 分钟）：{a.description[:30]}" for a in d.attractions]
            return f"第{day}天景点：\n" + "\n".join(lines)

        return f"第{day}天（{d.date}）：{d.description}"

    return "请问您想了解行程的哪部分？可以询问天气、预算、各天的景点、酒店或餐饮安排。"


async def query_handler_node(state: SupervisorState) -> dict:
    messages = state.get("messages", [])
    user_message = messages[-1].content if messages else ""
    reply = _build_query_reply(user_message, state)
    return {"messages": [AIMessage(content=reply)]}


async def other_handler_node(state: SupervisorState) -> dict:
    return {"messages": [AIMessage(
        content="我是行程助手，只能帮您查询或修改当前行程。请告诉我您想了解或修改什么？"
    )]}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_chat_graph_nodes.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/chat_graph.py backend/tests/test_chat_graph_nodes.py
git commit -m "feat: add query_handler and other_handler nodes"
```

---

## Task 4: `chat_graph.py` — modify_handler + classify_intent + graph 组装

**Files:**
- Modify: `backend/app/agents/chat_graph.py`
- Modify: `backend/tests/test_chat_graph_nodes.py`

- [ ] **Step 1: 补充 modify_handler 测试**

在 `backend/tests/test_chat_graph_nodes.py` 末尾追加：

```python
@pytest.mark.asyncio
async def test_modify_handler_calls_llm_and_returns_updated_plan():
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.agents.chat_graph import modify_handler_node

    mock_response = MagicMock()
    mock_response.content = '{"reply": "已修改行程", "updated_plan": null}'
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("app.agents.chat_graph.get_agent_llm", return_value=mock_llm):
        state = _make_state("帮我改轻松一点")
        result = await modify_handler_node(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "已修改行程"


@pytest.mark.asyncio
async def test_classify_intent_node_routes_to_query():
    from unittest.mock import AsyncMock, patch
    from app.agents.chat_graph import classify_intent_node

    with patch("app.agents.chat_graph.classify_intent", AsyncMock(return_value="query_plan")):
        state = _make_state("第一天住哪")
        cmd = await classify_intent_node(state)

    assert cmd.goto == "query_handler"


@pytest.mark.asyncio
async def test_classify_intent_node_routes_to_modify():
    from unittest.mock import AsyncMock, patch
    from app.agents.chat_graph import classify_intent_node

    with patch("app.agents.chat_graph.classify_intent", AsyncMock(return_value="modify")):
        state = _make_state("帮我改一下")
        cmd = await classify_intent_node(state)

    assert cmd.goto == "modify_handler"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_chat_graph_nodes.py::test_modify_handler_calls_llm_and_returns_updated_plan tests/test_chat_graph_nodes.py::test_classify_intent_node_routes_to_query -v
```

Expected: FAIL（`classify_intent_node` 和 `modify_handler_node` 未定义）

- [ ] **Step 3: 在 `chat_graph.py` 末尾追加 modify_handler + classify_intent + graph**

在 `backend/app/agents/chat_graph.py` 的 `other_handler_node` 函数后追加：

```python
async def modify_handler_node(state: SupervisorState) -> dict:
    from app.agents.llm_router import get_agent_llm
    from app.agents.state_trimmer import trim_state

    trimmed = trim_state(state, get_agent_llm("state_trimmer"))
    summary_ctx = f"历史摘要：{trimmed['summary']}\n" if trimmed.get("summary") else ""
    current_plan = state["trip_plan"].model_dump_json() if state.get("trip_plan") else "无"

    llm = get_agent_llm("modify_handler")
    prompt = [
        SystemMessage(content=(
            f"你是旅行修改助手。{summary_ctx}"
            f"当前行程 JSON：{current_plan}\n"
            '根据用户请求修改行程，返回 JSON：{"reply":"...","updated_plan":{...}}。'
            "如无需修改行程结构只需口头回答，updated_plan 返回原值。只返回 JSON。"
        )),
        *trimmed["messages"],
    ]

    response = await llm.ainvoke(prompt)

    try:
        data = json.loads(response.content)
        reply = data.get("reply", response.content)
        updated_plan_data = data.get("updated_plan")
        updated_plan = TripPlan(**updated_plan_data) if updated_plan_data else state.get("trip_plan")
    except Exception:
        reply = response.content
        updated_plan = state.get("trip_plan")

    update: dict = {
        "messages": [AIMessage(content=reply)],
        "trip_plan": updated_plan,
    }
    if trimmed.get("summary") != state.get("summary"):
        update["summary"] = trimmed["summary"]
    return update


async def classify_intent_node(state: SupervisorState) -> Command:
    messages = state.get("messages", [])
    if not messages:
        return Command(goto="other_handler")
    intent = await classify_intent(messages[-1].content)
    return Command(goto=_INTENT_TO_NODE[intent])


def create_chat_graph(checkpointer=None):
    builder = StateGraph(SupervisorState)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("query_handler", query_handler_node)
    builder.add_node("modify_handler", modify_handler_node)
    builder.add_node("other_handler", other_handler_node)
    builder.add_edge(START, "classify_intent")
    builder.add_edge("query_handler", END)
    builder.add_edge("modify_handler", END)
    builder.add_edge("other_handler", END)
    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: 运行全部节点测试**

```bash
pytest tests/test_chat_graph_nodes.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/chat_graph.py backend/tests/test_chat_graph_nodes.py
git commit -m "feat: add modify_handler, classify_intent node, and create_chat_graph()"
```

---

## Task 5: 接入 `__init__.py` + `main.py`

**Files:**
- Modify: `backend/app/agents/__init__.py`
- Modify: `backend/app/api/main.py`

- [ ] **Step 1: 更新 `backend/app/agents/__init__.py`**

完整替换为：

```python
from app.agents.supervisor import create_supervisor_graph
from app.agents.chat_graph import create_chat_graph

_supervisor_graph = None
_chat_graph = None


def init_supervisor_graph(checkpointer) -> None:
    global _supervisor_graph
    _supervisor_graph = create_supervisor_graph(checkpointer)


def get_supervisor_graph():
    if _supervisor_graph is None:
        raise RuntimeError("supervisor_graph not initialized")
    return _supervisor_graph


def init_chat_graph(checkpointer) -> None:
    global _chat_graph
    _chat_graph = create_chat_graph(checkpointer)


def get_chat_graph():
    if _chat_graph is None:
        raise RuntimeError("chat_graph not initialized")
    return _chat_graph
```

- [ ] **Step 2: 更新 `backend/app/api/main.py` startup**

将 startup 中的 graph 初始化部分替换为：

```python
    from ..services.checkpointer import init_checkpointer
    from ..agents import init_supervisor_graph, init_chat_graph
    checkpointer = await init_checkpointer()
    init_supervisor_graph(checkpointer)
    init_chat_graph(checkpointer)
```

（完整 startup 函数体，只改最后 3 行的 graph 初始化部分，其余不变）

- [ ] **Step 3: 启动应用确认无报错**

```bash
cd backend
python -m uvicorn app.api.main:app --port 8000
```

Expected: 控制台出现 `✅ Redis checkpointer 初始化成功`，无 ImportError

Ctrl+C 停止。

- [ ] **Step 4: Commit**

```bash
git add backend/app/agents/__init__.py backend/app/api/main.py
git commit -m "feat: register chat_graph in agents init and startup"
```

---

## Task 6: 简化 `chat.py`

**Files:**
- Modify: `backend/app/api/routes/chat.py`

- [ ] **Step 1: 完整替换 `backend/app/api/routes/chat.py`**

```python
"""多轮对话修改行程 API"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.agents import get_supervisor_graph, get_chat_graph
from app.models.schemas import TripPlan

router = APIRouter(prefix="/chat", tags=["多轮对话"])


class ChatModifyRequest(BaseModel):
    user_id: str
    message: str


class ChatModifyResponse(BaseModel):
    reply: str
    updated_plan: TripPlan | None = None


@router.get("/history/{user_id}", summary="获取对话历史")
async def get_history(user_id: str):
    graph = get_supervisor_graph()
    config = {"configurable": {"thread_id": user_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        return {"messages": []}
    msgs = [
        {"role": "user" if m.type == "human" else "ai", "content": m.content}
        for m in snapshot.values.get("messages", [])
    ]
    return {"messages": msgs}


@router.post("/modify", response_model=ChatModifyResponse, summary="多轮修改行程")
async def modify_trip(request: ChatModifyRequest):
    config = {"configurable": {"thread_id": request.user_id}}

    snapshot = await get_supervisor_graph().aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在，请先生成行程")

    result = await get_chat_graph().ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )

    messages = result.get("messages", [])
    reply = messages[-1].content if messages else ""
    return ChatModifyResponse(reply=reply, updated_plan=result.get("trip_plan"))
```

- [ ] **Step 2: 启动应用，手动测试完整流程**

先确保 Redis 在运行，然后：

```bash
python -m uvicorn app.api.main:app --port 8000 --reload
```

用 curl 或 Swagger UI (`http://localhost:8000/docs`) 测试：

1. POST `/api/trip/plan` 生成行程
2. GET `/api/chat/history/{user_id}` 确认消息存在
3. POST `/api/chat/modify` 发 "第一天住哪个酒店" → 应直接返回酒店名，无 LLM 延迟
4. POST `/api/chat/modify` 发 "谢谢" → 返回固定引导文案
5. POST `/api/chat/modify` 发 "帮我把第一天改得轻松一点" → 触发 LLM 修改

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/routes/chat.py
git commit -m "feat: simplify modify_trip to use chat_graph, intent routing active"
```
