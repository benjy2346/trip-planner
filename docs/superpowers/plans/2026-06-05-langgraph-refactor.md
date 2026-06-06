# LangGraph 重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph/LangChain 重构后端 Agent 系统，实现 Multi-Agent Supervisor 工作流、State 裁剪、LLM 降级路由、并行 SubGraph 分发，并新增 Redis 会话存储和多轮对话修改接口。

**Architecture:** SupervisorGraph 主图通过 LangGraph `Send()` API 并行分发到 WeatherSubGraph / HotelSubGraph / POISubGraph 三个独立子图，各子图状态完全隔离；主图状态由 `state_trimmer` 管理滑动窗口；会话状态持久化到 Redis（滑动 TTL 24h）。

**Tech Stack:** Python 3.11 / FastAPI / Pydantic v2 / asyncio / LangChain 0.3 / LangGraph 0.2 / langchain-mcp-adapters / redis[asyncio] / Vue 3

---

## File Map

**新建：**
- `backend/app/agents/state.py` — SupervisorState + 三个 SubState TypedDict
- `backend/app/agents/llm_router.py` — DeepSeek→Gemini→OpenAI 降级
- `backend/app/agents/state_trimmer.py` — 滑动窗口 + 摘要
- `backend/app/agents/subgraphs/__init__.py`
- `backend/app/agents/subgraphs/weather.py` — WeatherSubGraph
- `backend/app/agents/subgraphs/hotel.py` — HotelSubGraph
- `backend/app/agents/subgraphs/poi.py` — POISubGraph
- `backend/app/agents/supervisor.py` — SupervisorGraph 主图
- `backend/app/services/amap_tools.py` — langchain-mcp-adapters 包装
- `backend/app/services/session_store.py` — Redis 会话 CRUD
- `backend/app/api/routes/chat.py` — POST /api/chat/modify
- `backend/tests/__init__.py`
- `backend/tests/conftest.py`
- `backend/tests/test_llm_router.py`
- `backend/tests/test_state_trimmer.py`
- `backend/tests/test_supervisor.py`
- `backend/tests/test_session_store.py`
- `frontend/src/components/TripModifyChat.vue`

**修改：**
- `backend/requirements.txt` — 换包
- `backend/app/config.py` — 新增 REDIS_URL / DEEPSEEK_BASE_URL 等字段
- `backend/app/models/schemas.py` — TripRequest 加 user_id 字段
- `backend/app/agents/__init__.py` — 导出 supervisor_graph
- `backend/app/api/main.py` — 注册 chat 路由，startup 初始化 amap/redis
- `backend/app/api/routes/trip.py` — 改调用 SupervisorGraph，传 user_id

---

## Task 1: 更新依赖与配置

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py`
- Modify: `backend/app/models/schemas.py`

- [ ] **Step 1: 替换 requirements.txt**

```
# LangChain / LangGraph
langchain>=0.3.0
langchain-openai>=0.2.0
langgraph>=0.2.0
langchain-mcp-adapters>=0.1.0

# FastAPI 及相关
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Redis
redis[asyncio]>=5.0.0

# HTTP 客户端
httpx>=0.27.0
aiohttp>=3.10.0

# 工具
python-dotenv>=1.0.0
python-multipart>=0.0.9
loguru>=0.7.0
python-dateutil>=2.8.2
pytest>=8.0.0
pytest-asyncio>=0.23.0
fakeredis[aioredis]>=2.20.0
```

- [ ] **Step 2: 更新 config.py，新增所需字段**

在 `Settings` 类中新增（`openai_api_key` 等已有字段保留）：

```python
# DeepSeek
deepseek_api_key: str = ""
deepseek_base_url: str = "https://api.deepseek.com/v1"
deepseek_model: str = "deepseek-chat"

# Gemini（占位）
gemini_api_key: str = "placeholder"
gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
gemini_model: str = "gemini-1.5-flash"

# OpenAI（占位）
openai_model: str = "gpt-4o"

# Redis
redis_url: str = "redis://localhost:6379/0"
```

- [ ] **Step 3: 给 TripRequest 加 user_id 字段（schemas.py 第 10 行附近）**

```python
from uuid import uuid4

class TripRequest(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid4()), description="用户唯一标识")
    city: str = Field(..., description="目的地城市", example="北京")
    # ... 其余字段不变
```

- [ ] **Step 4: 安装依赖**

```bash
cd backend
pip install -r requirements.txt
```

- [ ] **Step 5: 验证安装**

```bash
python -c "import langgraph; import langchain_openai; import redis.asyncio; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/app/config.py backend/app/models/schemas.py
git commit -m "chore: replace hello-agents with langchain/langgraph stack"
```

---

## Task 2: State 定义

**Files:**
- Create: `backend/app/agents/state.py`
- Modify: `backend/app/agents/__init__.py`

- [ ] **Step 1: 创建 state.py**

```python
import operator
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from app.models.schemas import TripRequest, TripPlan, WeatherInfo, Hotel, Attraction


class SupervisorState(TypedDict):
    trip_request: TripRequest
    messages: Annotated[list, add_messages]
    trip_plan: Optional[TripPlan]
    summary: str
    weather_outputs: Annotated[list, operator.add]
    hotel_outputs: Annotated[list, operator.add]
    poi_outputs: Annotated[list, operator.add]


class WeatherSubState(TypedDict):
    city: str
    travel_dates: list[str]
    raw_result: str
    weather_result: list[WeatherInfo]


class HotelSubState(TypedDict):
    city: str
    accommodation_pref: str
    budget_level: str
    raw_result: str
    hotel_result: list[Hotel]


class POISubState(TypedDict):
    city: str
    preferences: list[str]
    travel_days: int
    raw_result: str
    poi_result: list[Attraction]
```

- [ ] **Step 2: 快速验证导入**

```bash
cd backend
python -c "from app.agents.state import SupervisorState, WeatherSubState; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/agents/state.py backend/app/agents/__init__.py
git commit -m "feat: add LangGraph state definitions"
```

---

## Task 3: LLM Router

**Files:**
- Create: `backend/app/agents/llm_router.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_llm_router.py`

- [ ] **Step 1: 写失败测试（test_llm_router.py）**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from openai import APITimeoutError, RateLimitError


@pytest.mark.asyncio
async def test_acall_uses_first_provider_when_healthy():
    fast = MagicMock()
    fast.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
    slow = MagicMock()

    from unittest.mock import patch
    with patch("app.agents.llm_router._build_providers", return_value=[fast, slow]):
        from app.agents.llm_router import acall_with_fallback
        result = await acall_with_fallback([HumanMessage(content="hi")])

    assert result.content == "ok"
    slow.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_acall_falls_back_on_timeout():
    failing = MagicMock()
    failing.ainvoke = AsyncMock(side_effect=APITimeoutError())
    working = MagicMock()
    working.ainvoke = AsyncMock(return_value=AIMessage(content="fallback"))

    from unittest.mock import patch
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

    from unittest.mock import patch
    with patch("app.agents.llm_router._build_providers", return_value=[failing, working]):
        from app.agents.llm_router import acall_with_fallback
        result = await acall_with_fallback([HumanMessage(content="hi")])

    assert result.content == "ok"


@pytest.mark.asyncio
async def test_acall_raises_when_all_fail():
    failing = MagicMock()
    failing.ainvoke = AsyncMock(side_effect=APITimeoutError())

    from unittest.mock import patch
    with patch("app.agents.llm_router._build_providers", return_value=[failing, failing]):
        from app.agents.llm_router import acall_with_fallback
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await acall_with_fallback([HumanMessage(content="hi")])
```

- [ ] **Step 2: 创建 tests/conftest.py**

```python
import pytest

pytest_plugins = ["pytest_asyncio"]
```

- [ ] **Step 3: 创建 tests/__init__.py（空文件）**

- [ ] **Step 4: 运行测试，确认失败**

```bash
cd backend
pytest tests/test_llm_router.py -v
```

Expected: `ImportError` 或 `ModuleNotFoundError`（llm_router 还不存在）

- [ ] **Step 5: 实现 llm_router.py**

```python
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
            base_url="https://api.openai.com/v1",
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
```

- [ ] **Step 6: 运行测试，确认通过**

```bash
cd backend
pytest tests/test_llm_router.py -v
```

Expected: 4 PASSED

- [ ] **Step 7: Commit**

```bash
git add backend/app/agents/llm_router.py backend/tests/
git commit -m "feat: add LLM router with DeepSeek→Gemini→OpenAI fallback"
```

---

## Task 4: Amap MCP Tools 集成

**Files:**
- Create: `backend/app/services/amap_tools.py`
- Modify: `backend/app/api/main.py`

- [ ] **Step 1: 创建 amap_tools.py**

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from app.config import get_settings

_client: MultiServerMCPClient | None = None
_tools: list[BaseTool] = []


async def init_amap_tools() -> None:
    global _client, _tools
    s = get_settings()
    if not s.amap_api_key:
        print("⚠️  AMAP_API_KEY 未配置，Amap 工具不可用")
        return
    _client = MultiServerMCPClient({
        "amap": {
            "command": "uvx",
            "args": ["amap-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": s.amap_api_key},
            "transport": "stdio",
        }
    })
    await _client.__aenter__()
    _tools = _client.get_tools()
    print(f"✅ Amap MCP 工具初始化成功，共 {len(_tools)} 个工具")


async def close_amap_tools() -> None:
    global _client
    if _client:
        await _client.__aexit__(None, None, None)
        _client = None


def get_amap_tools() -> list[BaseTool]:
    return _tools


def get_tool_by_name(name: str) -> BaseTool | None:
    return next((t for t in _tools if name in t.name), None)
```

- [ ] **Step 2: 更新 main.py 的 startup/shutdown**

在 `startup_event` 中添加 Amap 初始化（替换旧的 validate_config 部分，保留其余逻辑）：

```python
from app.services.amap_tools import init_amap_tools, close_amap_tools

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print("="*60)
    print_config()
    try:
        validate_config()
        print("\n✅ 配置验证通过")
    except ValueError as e:
        print(f"\n❌ 配置验证失败:\n{e}")
        raise
    await init_amap_tools()
    print("\n" + "="*60)
    print("📚 API文档: http://localhost:8000/docs")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    await close_amap_tools()
    print("\n👋 应用已关闭")
```

- [ ] **Step 3: 验证导入正确**

```bash
cd backend
python -c "from app.services.amap_tools import init_amap_tools, get_amap_tools; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/amap_tools.py backend/app/api/main.py
git commit -m "feat: wrap Amap MCP server as LangChain tools via langchain-mcp-adapters"
```

---

## Task 5: WeatherSubGraph

**Files:**
- Create: `backend/app/agents/subgraphs/__init__.py`
- Create: `backend/app/agents/subgraphs/weather.py`
- Create: `backend/tests/test_weather_subgraph.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_weather_subgraph.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.state import WeatherSubState


@pytest.mark.asyncio
async def test_weather_subgraph_returns_weather_info():
    mock_tool = MagicMock()
    mock_tool.name = "maps_weather"
    mock_tool.ainvoke = AsyncMock(return_value='{"status":"1","forecasts":[{"city":"北京"}]}')

    llm_response = MagicMock(content='[{"date":"2025-06-01","day_weather":"晴","night_weather":"多云","day_temp":28,"night_temp":18,"wind_direction":"南","wind_power":"3级"}]')

    with patch("app.agents.subgraphs.weather.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.weather.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):
            from app.agents.subgraphs.weather import weather_subgraph
            result = await weather_subgraph.ainvoke(
                WeatherSubState(city="北京", travel_dates=["2025-06-01"], raw_result="", weather_result=[])
            )

    assert len(result["weather_result"]) == 1
    assert result["weather_result"][0].day_weather == "晴"


@pytest.mark.asyncio
async def test_weather_subgraph_handles_parse_error():
    mock_tool = MagicMock()
    mock_tool.name = "maps_weather"
    mock_tool.ainvoke = AsyncMock(return_value="error response")

    llm_response = MagicMock(content="invalid json {{")

    with patch("app.agents.subgraphs.weather.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.weather.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):
            from app.agents.subgraphs.weather import weather_subgraph
            result = await weather_subgraph.ainvoke(
                WeatherSubState(city="北京", travel_dates=["2025-06-01"], raw_result="", weather_result=[])
            )

    assert result["weather_result"] == []
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd backend
pytest tests/test_weather_subgraph.py -v
```

Expected: `ImportError`（文件不存在）

- [ ] **Step 3: 创建 subgraphs/__init__.py（空文件）**

- [ ] **Step 4: 创建 subgraphs/weather.py**

```python
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import WeatherSubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import acall_with_fallback
from app.models.schemas import WeatherInfo


async def fetch_weather(state: WeatherSubState) -> dict:
    tool = next((t for t in get_amap_tools() if "weather" in t.name.lower()), None)
    if tool is None:
        return {"raw_result": "[]"}
    result = await tool.ainvoke({"city": state["city"]})
    return {"raw_result": str(result)}


async def parse_weather(state: WeatherSubState) -> dict:
    prompt = [
        SystemMessage(content=(
            "从高德天气查询结果中提取天气数据，返回 JSON 数组。\n"
            '每项格式：{"date":"YYYY-MM-DD","day_weather":"晴","night_weather":"多云",'
            '"day_temp":28,"night_temp":18,"wind_direction":"南","wind_power":"3级"}\n'
            "只返回 JSON 数组，不要其他文字。"
        )),
        HumanMessage(content=(
            f"查询结果：{state['raw_result']}\n"
            f"目标日期：{state['travel_dates']}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    try:
        data = json.loads(response.content)
        return {"weather_result": [WeatherInfo(**item) for item in data]}
    except Exception:
        return {"weather_result": []}


def create_weather_subgraph():
    g = StateGraph(WeatherSubState)
    g.add_node("fetch", fetch_weather)
    g.add_node("parse", parse_weather)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


weather_subgraph = create_weather_subgraph()
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
cd backend
pytest tests/test_weather_subgraph.py -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add backend/app/agents/subgraphs/ backend/tests/test_weather_subgraph.py
git commit -m "feat: add WeatherSubGraph with Amap MCP fetch + LLM parse"
```

---

## Task 6: HotelSubGraph + POISubGraph

**Files:**
- Create: `backend/app/agents/subgraphs/hotel.py`
- Create: `backend/app/agents/subgraphs/poi.py`
- Create: `backend/tests/test_hotel_poi_subgraph.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hotel_poi_subgraph.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.state import HotelSubState, POISubState


@pytest.mark.asyncio
async def test_hotel_subgraph_returns_hotel_list():
    mock_tool = MagicMock()
    mock_tool.name = "maps_text_search"
    mock_tool.ainvoke = AsyncMock(return_value='{"pois":[{"name":"如家酒店"}]}')

    llm_response = MagicMock(content='[{"name":"如家酒店","address":"北京市朝阳区","price_range":"200-300元/晚","rating":"4.2","distance":"市中心","type":"经济型","estimated_cost":250}]')

    with patch("app.agents.subgraphs.hotel.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.hotel.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):
            from app.agents.subgraphs.hotel import hotel_subgraph
            result = await hotel_subgraph.ainvoke(
                HotelSubState(city="北京", accommodation_pref="经济型", budget_level="mid", raw_result="", hotel_result=[])
            )

    assert len(result["hotel_result"]) == 1
    assert result["hotel_result"][0].name == "如家酒店"


@pytest.mark.asyncio
async def test_poi_subgraph_returns_attraction_list():
    mock_tool = MagicMock()
    mock_tool.name = "maps_text_search"
    mock_tool.ainvoke = AsyncMock(return_value='{"pois":[{"name":"故宫"}]}')

    llm_response = MagicMock(content='[{"name":"故宫","address":"北京市东城区景山前街4号","location":{"longitude":116.397,"latitude":39.916},"visit_duration":180,"description":"世界文化遗产","category":"历史文化","rating":4.8}]')

    with patch("app.agents.subgraphs.poi.get_amap_tools", return_value=[mock_tool]):
        with patch("app.agents.subgraphs.poi.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):
            from app.agents.subgraphs.poi import poi_subgraph
            result = await poi_subgraph.ainvoke(
                POISubState(city="北京", preferences=["历史文化"], travel_days=3, raw_result="", poi_result=[])
            )

    assert len(result["poi_result"]) >= 1
    assert result["poi_result"][0].name == "故宫"
```

- [ ] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/test_hotel_poi_subgraph.py -v
```

Expected: `ImportError`

- [ ] **Step 3: 创建 subgraphs/hotel.py**

```python
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import HotelSubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import acall_with_fallback
from app.models.schemas import Hotel


async def fetch_hotels(state: HotelSubState) -> dict:
    tool = next((t for t in get_amap_tools() if "search" in t.name.lower()), None)
    if tool is None:
        return {"raw_result": "[]"}
    result = await tool.ainvoke({
        "keywords": f"{state['accommodation_pref']}酒店",
        "city": state["city"],
    })
    return {"raw_result": str(result)}


async def parse_hotels(state: HotelSubState) -> dict:
    prompt = [
        SystemMessage(content=(
            "从高德 POI 搜索结果中提取酒店信息，返回 JSON 数组（最多 3 家）。\n"
            '每项格式：{"name":"...","address":"...","price_range":"...","rating":"...","distance":"...","type":"...","estimated_cost":0}\n'
            "只返回 JSON 数组。"
        )),
        HumanMessage(content=(
            f"搜索结果：{state['raw_result']}\n"
            f"偏好：{state['accommodation_pref']}，城市：{state['city']}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    try:
        data = json.loads(response.content)
        return {"hotel_result": [Hotel(**item) for item in data]}
    except Exception:
        return {"hotel_result": []}


def create_hotel_subgraph():
    g = StateGraph(HotelSubState)
    g.add_node("fetch", fetch_hotels)
    g.add_node("parse", parse_hotels)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


hotel_subgraph = create_hotel_subgraph()
```

- [ ] **Step 4: 创建 subgraphs/poi.py**

```python
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import POISubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import acall_with_fallback
from app.models.schemas import Attraction, Location


async def fetch_pois(state: POISubState) -> dict:
    tool = next((t for t in get_amap_tools() if "search" in t.name.lower()), None)
    if tool is None:
        return {"raw_result": "[]"}
    keywords = "、".join(state["preferences"]) if state["preferences"] else "热门景点"
    result = await tool.ainvoke({"keywords": keywords, "city": state["city"]})
    return {"raw_result": str(result)}


async def parse_pois(state: POISubState) -> dict:
    prompt = [
        SystemMessage(content=(
            "从高德 POI 搜索结果中提取景点信息，返回 JSON 数组。\n"
            '每项格式：{"name":"...","address":"...","location":{"longitude":0.0,"latitude":0.0},'
            '"visit_duration":120,"description":"...","category":"...","rating":4.5,"ticket_price":0}\n'
            "只返回 JSON 数组。"
        )),
        HumanMessage(content=(
            f"搜索结果：{state['raw_result']}\n"
            f"偏好：{state['preferences']}，天数：{state['travel_days']}，城市：{state['city']}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    try:
        data = json.loads(response.content)
        attractions = []
        for item in data:
            if "location" not in item:
                item["location"] = {"longitude": 0.0, "latitude": 0.0}
            attractions.append(Attraction(**item))
        return {"poi_result": attractions}
    except Exception:
        return {"poi_result": []}


def create_poi_subgraph():
    g = StateGraph(POISubState)
    g.add_node("fetch", fetch_pois)
    g.add_node("parse", parse_pois)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


poi_subgraph = create_poi_subgraph()
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
cd backend
pytest tests/test_hotel_poi_subgraph.py -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add backend/app/agents/subgraphs/hotel.py backend/app/agents/subgraphs/poi.py backend/tests/test_hotel_poi_subgraph.py
git commit -m "feat: add HotelSubGraph and POISubGraph"
```

---

## Task 7: State Trimmer

**Files:**
- Create: `backend/app/agents/state_trimmer.py`
- Create: `backend/tests/test_state_trimmer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_state_trimmer.py
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.state_trimmer import trim_state, WINDOW_SIZE


def _make_state(n: int, trip_plan=None, summary=""):
    msgs = [HumanMessage(content=f"msg {i}") for i in range(n)]
    return {
        "messages": msgs,
        "trip_plan": trip_plan,
        "summary": summary,
        "trip_request": None,
        "weather_outputs": [],
        "hotel_outputs": [],
        "poi_outputs": [],
    }


def test_no_trim_when_under_window():
    state = _make_state(3)
    mock_llm = MagicMock()
    result = trim_state(state, mock_llm)
    assert len(result["messages"]) == 3
    mock_llm.invoke.assert_not_called()


def test_trim_keeps_window_size_messages():
    state = _make_state(12)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="摘要")
    result = trim_state(state, mock_llm)
    assert len(result["messages"]) == WINDOW_SIZE


def test_trip_plan_preserved_after_trim():
    sentinel = {"city": "Beijing"}
    state = _make_state(12, trip_plan=sentinel)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="摘要")
    result = trim_state(state, mock_llm)
    assert result["trip_plan"] is sentinel


def test_summary_updated_after_trim():
    state = _make_state(12, summary="旧摘要")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="新摘要")
    result = trim_state(state, mock_llm)
    assert result["summary"] == "新摘要"


def test_no_trim_exactly_at_window():
    state = _make_state(WINDOW_SIZE)
    mock_llm = MagicMock()
    result = trim_state(state, mock_llm)
    assert len(result["messages"]) == WINDOW_SIZE
    mock_llm.invoke.assert_not_called()
```

- [ ] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/test_state_trimmer.py -v
```

Expected: `ImportError`

- [ ] **Step 3: 实现 state_trimmer.py**

```python
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

WINDOW_SIZE = 4


def trim_state(state: dict, llm) -> dict:
    messages: list[BaseMessage] = state.get("messages", [])
    if len(messages) <= WINDOW_SIZE:
        return state

    to_summarize = messages[:-WINDOW_SIZE]
    kept = messages[-WINDOW_SIZE:]
    existing_summary = state.get("summary", "")

    summary_prompt = [
        SystemMessage(content=(
            "请将以下对话历史压缩为简洁摘要，保留关键行程偏好和修改意图，"
            "丢弃无关闲聊。摘要用中文，不超过 200 字。"
        )),
        HumanMessage(content=(
            f"已有摘要：{existing_summary}\n\n"
            "新增对话：\n" +
            "\n".join(f"{m.type}: {m.content}" for m in to_summarize)
        )),
    ]
    new_summary = llm.invoke(summary_prompt)

    return {
        **state,
        "messages": kept,
        "summary": new_summary.content,
    }
```

- [ ] **Step 4: 运行，确认通过**

```bash
cd backend
pytest tests/test_state_trimmer.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/state_trimmer.py backend/tests/test_state_trimmer.py
git commit -m "feat: add State trimmer with sliding window and dynamic summarization"
```

---

## Task 8: SupervisorGraph

**Files:**
- Create: `backend/app/agents/supervisor.py`
- Modify: `backend/app/agents/__init__.py`
- Create: `backend/tests/test_supervisor.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_supervisor.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage
from app.models.schemas import TripRequest, TripPlan, WeatherInfo, Hotel, Attraction, Location


def _make_request():
    return TripRequest(
        user_id="test-user",
        city="北京",
        start_date="2025-06-01",
        end_date="2025-06-03",
        travel_days=3,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )


@pytest.mark.asyncio
async def test_supervisor_returns_trip_plan():
    weather_out = [WeatherInfo(date="2025-06-01", day_weather="晴", night_weather="多云", day_temp=28, night_temp=18)]
    hotel_out = [Hotel(name="如家", address="北京朝阳", price_range="200-300", rating="4.2", type="经济型")]
    poi_out = [Attraction(name="故宫", address="东城区", location=Location(longitude=116.4, latitude=39.9), visit_duration=180, description="历史")]

    mock_weather_result = {"weather_result": weather_out}
    mock_hotel_result = {"hotel_result": hotel_out}
    mock_poi_result = {"poi_result": poi_out}

    plan_json = '{"city":"北京","start_date":"2025-06-01","end_date":"2025-06-03","days":[],"overall_suggestions":"推荐早起"}'
    llm_response = MagicMock(content=plan_json)

    with patch("app.agents.supervisor.weather_subgraph") as mock_w, \
         patch("app.agents.supervisor.hotel_subgraph") as mock_h, \
         patch("app.agents.supervisor.poi_subgraph") as mock_p, \
         patch("app.agents.supervisor.acall_with_fallback", new_callable=AsyncMock, return_value=llm_response):

        mock_w.ainvoke = AsyncMock(return_value=mock_weather_result)
        mock_h.ainvoke = AsyncMock(return_value=mock_hotel_result)
        mock_p.ainvoke = AsyncMock(return_value=mock_poi_result)

        from app.agents.supervisor import supervisor_graph
        from app.agents.state import SupervisorState
        result = await supervisor_graph.ainvoke(SupervisorState(
            trip_request=_make_request(),
            messages=[],
            trip_plan=None,
            summary="",
            weather_outputs=[],
            hotel_outputs=[],
            poi_outputs=[],
        ))

    assert result["trip_plan"] is not None
    assert result["trip_plan"].city == "北京"


@pytest.mark.asyncio
async def test_all_three_subgraphs_invoked():
    mock_result = lambda out_key, items: {out_key: items}

    with patch("app.agents.supervisor.weather_subgraph") as mock_w, \
         patch("app.agents.supervisor.hotel_subgraph") as mock_h, \
         patch("app.agents.supervisor.poi_subgraph") as mock_p, \
         patch("app.agents.supervisor.acall_with_fallback", new_callable=AsyncMock,
               return_value=MagicMock(content='{"city":"北京","start_date":"2025-06-01","end_date":"2025-06-03","days":[],"overall_suggestions":"ok"}')):

        mock_w.ainvoke = AsyncMock(return_value={"weather_result": []})
        mock_h.ainvoke = AsyncMock(return_value={"hotel_result": []})
        mock_p.ainvoke = AsyncMock(return_value={"poi_result": []})

        from app.agents.supervisor import supervisor_graph
        from app.agents.state import SupervisorState
        await supervisor_graph.ainvoke(SupervisorState(
            trip_request=_make_request(),
            messages=[], trip_plan=None, summary="",
            weather_outputs=[], hotel_outputs=[], poi_outputs=[],
        ))

    mock_w.ainvoke.assert_called_once()
    mock_h.ainvoke.assert_called_once()
    mock_p.ainvoke.assert_called_once()
```

- [ ] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/test_supervisor.py -v
```

Expected: `ImportError`

- [ ] **Step 3: 实现 supervisor.py**

```python
import json
from datetime import date, timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.agents.state import SupervisorState, WeatherSubState, HotelSubState, POISubState
from app.agents.subgraphs.weather import weather_subgraph
from app.agents.subgraphs.hotel import hotel_subgraph
from app.agents.subgraphs.poi import poi_subgraph
from app.agents.llm_router import acall_with_fallback
from app.models.schemas import TripPlan


def _date_range(start: str, days: int) -> list[str]:
    d = date.fromisoformat(start)
    return [(d + timedelta(days=i)).isoformat() for i in range(days)]


def dispatch_subgraphs(state: SupervisorState) -> list[Send]:
    req = state["trip_request"]
    return [
        Send("run_weather", WeatherSubState(
            city=req.city,
            travel_dates=_date_range(req.start_date, req.travel_days),
            raw_result="",
            weather_result=[],
        )),
        Send("run_hotel", HotelSubState(
            city=req.city,
            accommodation_pref=req.accommodation,
            budget_level="mid",
            raw_result="",
            hotel_result=[],
        )),
        Send("run_poi", POISubState(
            city=req.city,
            preferences=req.preferences,
            travel_days=req.travel_days,
            raw_result="",
            poi_result=[],
        )),
    ]


async def run_weather_node(sub_state: WeatherSubState) -> dict:
    result = await weather_subgraph.ainvoke(sub_state)
    return {"weather_outputs": result["weather_result"]}


async def run_hotel_node(sub_state: HotelSubState) -> dict:
    result = await hotel_subgraph.ainvoke(sub_state)
    return {"hotel_outputs": result["hotel_result"]}


async def run_poi_node(sub_state: POISubState) -> dict:
    result = await poi_subgraph.ainvoke(sub_state)
    return {"poi_outputs": result["poi_result"]}


async def assembler_node(state: SupervisorState) -> dict:
    req = state["trip_request"]
    prompt = [
        SystemMessage(content=(
            "你是旅行规划助手。根据提供的天气、酒店、景点数据生成详细行程，返回 JSON。\n"
            '格式：{"city":"...","start_date":"...","end_date":"...","days":[...],"overall_suggestions":"..."}\n'
            "只返回 JSON，不要其他文字。"
        )),
        HumanMessage(content=(
            f"城市：{req.city}，{req.start_date}~{req.end_date}，{req.travel_days}天\n"
            f"交通：{req.transportation}，住宿：{req.accommodation}\n"
            f"偏好：{req.preferences}\n"
            f"{f'额外要求：{req.free_text_input}' if req.free_text_input else ''}\n\n"
            f"天气：{[w.model_dump() for w in state.get('weather_outputs', [])]}\n"
            f"酒店：{[h.model_dump() for h in state.get('hotel_outputs', [])]}\n"
            f"景点：{[p.model_dump() for p in state.get('poi_outputs', [])]}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    try:
        data = json.loads(response.content)
        trip_plan = TripPlan(**data)
    except Exception:
        trip_plan = TripPlan(
            city=req.city,
            start_date=req.start_date,
            end_date=req.end_date,
            days=[],
            overall_suggestions="行程生成失败，请重试",
        )
    return {
        "trip_plan": trip_plan,
        "messages": [AIMessage(content=f"已为您生成{req.city}{req.travel_days}天行程。")],
    }


def create_supervisor_graph():
    builder = StateGraph(SupervisorState)
    builder.add_node("run_weather", run_weather_node)
    builder.add_node("run_hotel", run_hotel_node)
    builder.add_node("run_poi", run_poi_node)
    builder.add_node("assembler", assembler_node)
    builder.add_conditional_edges(START, dispatch_subgraphs)
    builder.add_edge("run_weather", "assembler")
    builder.add_edge("run_hotel", "assembler")
    builder.add_edge("run_poi", "assembler")
    builder.add_edge("assembler", END)
    return builder.compile()


supervisor_graph = create_supervisor_graph()
```

- [ ] **Step 4: 更新 agents/__init__.py**

```python
from app.agents.supervisor import supervisor_graph

__all__ = ["supervisor_graph"]
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
cd backend
pytest tests/test_supervisor.py -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add backend/app/agents/supervisor.py backend/app/agents/__init__.py backend/tests/test_supervisor.py
git commit -m "feat: add SupervisorGraph with parallel Send() dispatch to three SubGraphs"
```

---

## Task 9: Redis Session Store + 更新 trip.py

**Files:**
- Create: `backend/app/services/session_store.py`
- Create: `backend/tests/test_session_store.py`
- Modify: `backend/app/api/main.py`
- Modify: `backend/app/api/routes/trip.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_session_store.py
import pytest
import fakeredis.aioredis
from unittest.mock import patch
from app.agents.state import SupervisorState
from app.models.schemas import TripRequest


def _make_state() -> SupervisorState:
    return SupervisorState(
        trip_request=TripRequest(
            user_id="u1", city="北京", start_date="2025-06-01", end_date="2025-06-03",
            travel_days=3, transportation="公共交通", accommodation="经济型",
        ),
        messages=[],
        trip_plan=None,
        summary="",
        weather_outputs=[],
        hotel_outputs=[],
        poi_outputs=[],
    )


@pytest.mark.asyncio
async def test_save_and_load_session():
    fake = fakeredis.aioredis.FakeRedis()
    with patch("app.services.session_store._get_redis", return_value=fake):
        from app.services.session_store import save_session, load_session
        state = _make_state()
        await save_session("u1", state)
        loaded = await load_session("u1")

    assert loaded is not None
    assert loaded["trip_request"].city == "北京"


@pytest.mark.asyncio
async def test_load_nonexistent_session_returns_none():
    fake = fakeredis.aioredis.FakeRedis()
    with patch("app.services.session_store._get_redis", return_value=fake):
        from app.services.session_store import load_session
        result = await load_session("nonexistent")

    assert result is None


@pytest.mark.asyncio
async def test_save_refreshes_ttl():
    fake = fakeredis.aioredis.FakeRedis()
    with patch("app.services.session_store._get_redis", return_value=fake):
        from app.services.session_store import save_session
        state = _make_state()
        await save_session("u1", state)
        ttl = await fake.ttl("session:u1")

    assert ttl > 0
```

- [ ] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/test_session_store.py -v
```

Expected: `ImportError`

- [ ] **Step 3: 实现 session_store.py**

```python
import json
from typing import Optional
import redis.asyncio as aioredis
from app.config import get_settings
from app.agents.state import SupervisorState
from app.models.schemas import TripRequest, TripPlan

SESSION_TTL = 86400  # 24h 滑动 TTL
_redis_client: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis_client


async def init_redis() -> None:
    client = _get_redis()
    await client.ping()
    print("✅ Redis 连接成功")


async def close_redis() -> None:
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None


def _serialize(state: SupervisorState) -> str:
    data = dict(state)
    data["trip_request"] = state["trip_request"].model_dump() if state.get("trip_request") else None
    data["trip_plan"] = state["trip_plan"].model_dump() if state.get("trip_plan") else None
    data["messages"] = [{"type": m.type, "content": m.content} for m in state.get("messages", [])]
    return json.dumps(data, ensure_ascii=False)


def _deserialize(raw: str) -> SupervisorState:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    _msg_map = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}
    data = json.loads(raw)
    if data.get("trip_request"):
        data["trip_request"] = TripRequest(**data["trip_request"])
    if data.get("trip_plan"):
        data["trip_plan"] = TripPlan(**data["trip_plan"])
    data["messages"] = [_msg_map.get(m["type"], HumanMessage)(content=m["content"]) for m in data.get("messages", [])]
    return SupervisorState(**data)


async def save_session(user_id: str, state: SupervisorState) -> None:
    client = _get_redis()
    await client.set(f"session:{user_id}", _serialize(state), ex=SESSION_TTL)


async def load_session(user_id: str) -> Optional[SupervisorState]:
    client = _get_redis()
    raw = await client.get(f"session:{user_id}")
    if raw is None:
        return None
    await client.expire(f"session:{user_id}", SESSION_TTL)
    return _deserialize(raw)
```

- [ ] **Step 4: 更新 main.py，添加 Redis 初始化和 chat 路由**

```python
# 在 main.py 顶部 import 区新增：
from app.services.session_store import init_redis, close_redis
from app.api.routes import chat as chat_routes

# startup_event 中 init_amap_tools() 后新增：
await init_redis()

# shutdown_event 中新增：
await close_redis()

# 路由注册区新增：
app.include_router(chat_routes.router, prefix="/api")
```

- [ ] **Step 5: 更新 trip.py，调用 SupervisorGraph**

完整替换 `backend/app/api/routes/trip.py` 内容：

```python
from fastapi import APIRouter, HTTPException
from app.models.schemas import TripRequest, TripPlanResponse
from app.agents import supervisor_graph
from app.agents.state import SupervisorState
from app.services.session_store import save_session

router = APIRouter(prefix="/trip", tags=["旅行规划"])


@router.post("/plan", response_model=TripPlanResponse, summary="生成旅行计划")
async def plan_trip(request: TripRequest):
    try:
        initial_state = SupervisorState(
            trip_request=request,
            messages=[],
            trip_plan=None,
            summary="",
            weather_outputs=[],
            hotel_outputs=[],
            poi_outputs=[],
        )
        result = await supervisor_graph.ainvoke(initial_state)
        await save_session(request.user_id, result)
        return TripPlanResponse(success=True, message="旅行计划生成成功", data=result["trip_plan"])
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {e}")


@router.get("/health", summary="健康检查")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-langgraph"}
```

- [ ] **Step 6: 运行测试**

```bash
cd backend
pytest tests/test_session_store.py -v
```

Expected: 3 PASSED

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/session_store.py backend/app/api/routes/trip.py backend/app/api/main.py backend/tests/test_session_store.py
git commit -m "feat: add Redis session store and wire SupervisorGraph to /trip/plan"
```

---

## Task 10: Chat 修改接口

**Files:**
- Create: `backend/app/api/routes/chat.py`

- [ ] **Step 1: 创建 chat.py**

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.llm_router import acall_with_fallback
from app.agents.state_trimmer import trim_state
from app.agents.llm_router import _build_providers
from app.services.session_store import load_session, save_session
from app.models.schemas import TripPlan
from langchain_core.messages import SystemMessage
import json

router = APIRouter(prefix="/chat", tags=["多轮对话"])


class ChatModifyRequest(BaseModel):
    user_id: str
    message: str


class ChatModifyResponse(BaseModel):
    reply: str
    updated_plan: TripPlan | None = None


@router.post("/modify", response_model=ChatModifyResponse, summary="多轮修改行程")
async def modify_trip(request: ChatModifyRequest):
    state = await load_session(request.user_id)
    if state is None:
        raise HTTPException(status_code=404, detail="会话不存在，请先生成行程")

    llm = _build_providers()[0]
    state = trim_state(state, llm)

    summary_ctx = f"历史摘要：{state['summary']}\n" if state.get("summary") else ""
    current_plan = state["trip_plan"].model_dump_json() if state.get("trip_plan") else "无"

    prompt = [
        SystemMessage(content=(
            f"你是旅行修改助手。{summary_ctx}"
            f"当前行程 JSON：{current_plan}\n"
            "根据用户请求修改行程，返回 JSON：{\"reply\":\"...\",\"updated_plan\":{...}} 。"
            "如无需修改行程结构只需口头回答，updated_plan 返回原值。只返回 JSON。"
        )),
        *state["messages"],
        HumanMessage(content=request.message),
    ]

    response = await acall_with_fallback(prompt)

    try:
        data = json.loads(response.content)
        reply = data.get("reply", response.content)
        updated_plan_data = data.get("updated_plan")
        updated_plan = TripPlan(**updated_plan_data) if updated_plan_data else state["trip_plan"]
    except Exception:
        reply = response.content
        updated_plan = state["trip_plan"]

    new_state = {
        **state,
        "messages": list(state["messages"]) + [
            HumanMessage(content=request.message),
            AIMessage(content=reply),
        ],
        "trip_plan": updated_plan,
    }
    await save_session(request.user_id, new_state)

    return ChatModifyResponse(reply=reply, updated_plan=updated_plan)
```

- [ ] **Step 2: 验证路由注册正常**

```bash
cd backend
python -c "from app.api.routes.chat import router; print(router.prefix)"
```

Expected: `/chat`

- [ ] **Step 3: 启动服务验证接口文档可访问**

```bash
cd backend
python run.py
# 浏览器打开 http://localhost:8000/docs
# 确认 /api/chat/modify 出现在文档中
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/api/routes/chat.py
git commit -m "feat: add /api/chat/modify endpoint for multi-turn trip modification"
```

---

## Task 11: 前端 TripModifyChat 组件

**Files:**
- Create: `frontend/src/components/TripModifyChat.vue`
- Modify: 行程结果展示页（找到 `src/views/` 或 `src/pages/` 下展示 TripPlan 的页面，加入 chat 组件）

- [ ] **Step 1: 创建 TripModifyChat.vue**

```vue
<template>
  <div class="chat-panel">
    <div class="chat-header">
      <span>💬 修改行程</span>
    </div>

    <div class="chat-messages" ref="messageContainer">
      <div
        v-for="(msg, i) in messages"
        :key="i"
        :class="['message', msg.role]"
      >
        <div class="bubble">{{ msg.content }}</div>
      </div>
      <div v-if="loading" class="message ai">
        <div class="bubble loading">思考中...</div>
      </div>
    </div>

    <div class="chat-input">
      <a-input
        v-model:value="inputText"
        placeholder="输入修改需求，例如：把第二天改成爬长城"
        :disabled="loading"
        @pressEnter="send"
      />
      <a-button type="primary" :loading="loading" @click="send">发送</a-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue'
import axios from 'axios'

const props = defineProps<{ userId: string }>()
const emit = defineEmits<{ (e: 'plan-updated', plan: unknown): void }>()

interface Message { role: 'user' | 'ai'; content: string }

const messages = ref<Message[]>([])
const inputText = ref('')
const loading = ref(false)
const messageContainer = ref<HTMLElement | null>(null)

async function send() {
  const text = inputText.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  loading.value = true
  await scrollToBottom()

  try {
    const { data } = await axios.post('/api/chat/modify', {
      user_id: props.userId,
      message: text,
    })
    messages.value.push({ role: 'ai', content: data.reply })
    if (data.updated_plan) {
      emit('plan-updated', data.updated_plan)
    }
  } catch (err) {
    messages.value.push({ role: 'ai', content: '修改失败，请重试' })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

async function scrollToBottom() {
  await nextTick()
  if (messageContainer.value) {
    messageContainer.value.scrollTop = messageContainer.value.scrollHeight
  }
}
</script>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  border-left: 1px solid #e8e8e8;
  background: #fafafa;
}
.chat-header {
  padding: 12px 16px;
  font-weight: 600;
  border-bottom: 1px solid #e8e8e8;
  background: #fff;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.message { display: flex; }
.message.user { justify-content: flex-end; }
.message.ai { justify-content: flex-start; }
.bubble {
  max-width: 80%;
  padding: 8px 12px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.5;
}
.message.user .bubble { background: #1890ff; color: #fff; border-radius: 12px 12px 0 12px; }
.message.ai .bubble { background: #fff; border: 1px solid #e8e8e8; border-radius: 12px 12px 12px 0; }
.bubble.loading { color: #999; font-style: italic; }
.chat-input {
  padding: 12px;
  display: flex;
  gap: 8px;
  border-top: 1px solid #e8e8e8;
  background: #fff;
}
.chat-input .ant-input { flex: 1; }
</style>
```

- [ ] **Step 2: 修改 `frontend/src/views/Result.vue`，引入 TripModifyChat**

在 `Result.vue` 的 `<template>` 最外层 `<div class="result-container">` 改为两列布局：

```vue
<template>
  <a-row :gutter="0" style="height: calc(100vh - 112px)">
    <!-- 左：行程展示（现有内容） -->
    <a-col :span="16" style="overflow-y: auto; padding: 24px">
      <!-- 现有行程渲染代码保持不变 -->
    </a-col>

    <!-- 右：修改 chat -->
    <a-col :span="8" style="height: 100%">
      <TripModifyChat
        :user-id="userId"
        @plan-updated="onPlanUpdated"
      />
    </a-col>
  </a-row>
</template>

<script setup lang="ts">
import TripModifyChat from '@/components/TripModifyChat.vue'
// userId 从 localStorage 读取：
const userId = localStorage.getItem('trip_user_id') ?? (() => {
  const id = crypto.randomUUID()
  localStorage.setItem('trip_user_id', id)
  return id
})()

function onPlanUpdated(newPlan: unknown) {
  // 更新当前页面的行程数据
  tripPlan.value = newPlan
}
</script>
```

- [ ] **Step 3: 启动前端，验证 chat 面板可见**

```bash
cd frontend
npm run dev
# 浏览器打开 http://localhost:5173
# 生成行程后确认右侧出现 chat 面板
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/TripModifyChat.vue
git commit -m "feat: add TripModifyChat Vue component for multi-turn itinerary modification"
```

---

## 完整测试运行

- [ ] **运行所有后端测试**

```bash
cd backend
pytest tests/ -v
```

Expected: 所有测试 PASSED

- [ ] **最终 commit**

```bash
git add .
git commit -m "feat: complete LangGraph refactor - Supervisor/SubGraph/Trimmer/Router/Redis"
```
