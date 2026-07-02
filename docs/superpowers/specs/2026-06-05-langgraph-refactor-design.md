# LangGraph 重构设计文档

**日期：** 2026-06-05  
**技术栈：** Python / FastAPI / Pydantic / asyncio / LangChain / LangGraph / Redis / Vue

---

## 背景

当前项目基于 HelloAgents 框架，agents 模块为空，核心 Agent 逻辑尚未实现。本次重构目标：

1. 用 LangGraph 实现 Multi-Agent Supervisor 工作流，对齐简历描述
2. 保留现有 FastAPI + Pydantic + Vue 技术栈，保留高德地图 MCP 集成
3. 新增多轮对话修改行程能力（双模式：表单生成 + chat 修改）
4. 引入 Redis 存储每个用户的对话历史

---

## 交互模式

**双模式设计：**

1. **表单模式（现有）**：用户填写城市、日期、偏好等结构化字段，提交后生成完整行程
2. **Chat 修改模式（新增）**：行程生成后，右侧出现 chat 抽屉，用户用自然语言修改行程（如"把第二天改成爬长城"）

用户标识：前端首次访问时在 `localStorage` 生成 UUID 作为 `user_id`，所有请求携带，不做注册登录。

---

## 目录结构

```
backend/app/
├── agents/
│   ├── supervisor.py          # SupervisorGraph 主图
│   ├── subgraphs/
│   │   ├── weather.py         # WeatherSubGraph
│   │   ├── hotel.py           # HotelSubGraph
│   │   └── poi.py             # POISubGraph
│   ├── state.py               # 所有 State TypedDict 定义
│   ├── state_trimmer.py       # 滑动窗口 + 动态摘要
│   └── llm_router.py          # DeepSeek→Gemini→OpenAI 降级链
├── api/routes/
│   ├── trip.py                # POST /api/trip/plan
│   └── chat.py                # POST /api/chat/modify（新增）
├── services/
│   ├── amap_tools.py          # Amap MCP → LangChain Tool 包装
│   ├── session_store.py       # Redis 会话存取封装
│   └── unsplash_service.py    # 保留不动
├── models/schemas.py          # 保留，Pydantic 模型不变
└── config.py                  # 新增 REDIS_URL、DeepSeek/Gemini key 字段
```

---

## State 定义

### 主图 State

```python
class SupervisorState(TypedDict):
    trip_request: TripRequest
    messages: Annotated[list, add_messages]   # 多轮对话历史，被 trimmer 管理
    trip_plan: Optional[TripPlan]             # 当前行程，独立存储不走消息压缩
    summary: str                              # trimmer 生成的历史摘要
```

### SubGraph State（各自独立，与主图完全隔离）

```python
class WeatherSubState(TypedDict):
    city: str
    travel_dates: list[str]
    weather_result: list[WeatherInfo]

class HotelSubState(TypedDict):
    city: str
    accommodation_pref: str
    budget_level: str
    hotel_result: list[Hotel]

class POISubState(TypedDict):
    city: str
    preferences: list[str]
    travel_days: int
    poi_result: list[Attraction]
```

`TripPlan` 始终存储在 `SupervisorState.trip_plan` 字段，不放入 `messages`，因此永远不会被摘要压缩。

---

## LangGraph 图结构

### SupervisorGraph

```
supervisor_node（意图分类 + 路由决策）
    │
    ├── Send("weather_subgraph", WeatherSubState)  ─┐
    ├── Send("hotel_subgraph",   HotelSubState)    ─┤ 并行执行
    └── Send("poi_subgraph",     POISubState)      ─┘
                                                    │
                                            assembler_node（合并结果 → TripPlan）
```

并行分发使用 LangGraph 原生 `Send()` API，无需手写 `asyncio.gather()`。

### 多轮修改路由

`supervisor_node` 对用户消息做意图分类后按需分发：

| 意图 | 触发的 SubGraph |
|------|---------------|
| 修改天气建议 | weather_subgraph |
| 更换酒店 | hotel_subgraph |
| 增删景点 | poi_subgraph |
| 重新规划 | 三个同时 Send |

### SubGraph 内部（以 WeatherSubGraph 为例）

```
fetch_node  → 调用 Amap MCP Tool（maps_weather）
parse_node  → LLM 将 raw 结果结构化为 WeatherInfo[]
```

每个 SubGraph 是独立的 `StateGraph(...).compile()`，状态完全隔离。

---

## Amap MCP 集成

用 `langchain-mcp-adapters` 把 `uvx amap-mcp-server` 包装为 LangChain Tool：

```python
# services/amap_tools.py
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_amap_tools():
    async with MultiServerMCPClient({
        "amap": {"command": "uvx", "args": ["amap-mcp-server"],
                 "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key}}
    }) as client:
        return client.get_tools()
# 包含：maps_weather, maps_text_search, maps_direction_walking, maps_geo 等
```

SubGraph 的 `fetch_node` 直接调用对应 tool，LLM 在 `parse_node` 解释结果并结构化，无需手动解析 JSON。

---

## State Trimmer

对应简历第二条：滑动窗口 + 动态摘要。

**触发条件：** `messages` 超过 10 条，或估算 token 数超过 3000

**裁剪流程：**

1. 保留最近 4 条消息（滑动窗口）
2. 对窗口外消息调用 LLM 生成摘要，合并追加到 `state.summary`
3. 下次调用时将 `summary` 注入 system prompt 开头
4. `trip_plan` 独立字段，不参与压缩

```python
WINDOW_SIZE = 4
TOKEN_THRESHOLD = 3000

def trim_state(state: SupervisorState) -> SupervisorState:
    if len(state["messages"]) <= WINDOW_SIZE:
        return state
    to_summarize = state["messages"][:-WINDOW_SIZE]
    new_summary = llm.invoke(summarize_prompt(state["summary"], to_summarize))
    return {
        **state,
        "messages": state["messages"][-WINDOW_SIZE:],
        "summary": new_summary.content,
    }
```

---

## LLM Router

对应简历第三条：跨供应商动态路由降级链路。

**优先级：** DeepSeek（主力）→ Gemini（占位）→ OpenAI（占位）

**实现：**

```python
PROVIDERS = [
    ChatOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_KEY, timeout=8),
    ChatOpenAI(base_url=GEMINI_BASE_URL,   api_key=GEMINI_KEY,   timeout=8),
    ChatOpenAI(base_url=OPENAI_BASE_URL,   api_key=OPENAI_KEY,   timeout=8),
]
```

在 SubGraph 的每次 LLM 调用处捕获异常并切换：

```python
for llm in PROVIDERS:
    try:
        return llm.invoke(messages)
    except (TimeoutError, RateLimitError, APIConnectionError):
        continue
raise RuntimeError("所有 LLM 供应商不可用")
```

捕获在调用点而非探测阶段，切换延迟在毫秒级。

---

## Redis 会话存储

**数据结构：**

```
key:   session:{user_id}
value: JSON 序列化的 SupervisorState
TTL:   86400 秒（24 小时）
```

**封装（`services/session_store.py`）：**

```python
async def save_session(user_id: str, state: SupervisorState) -> None: ...
async def load_session(user_id: str) -> SupervisorState | None: ...
```

每次 `/chat/modify` 请求：先从 Redis 加载历史 state → 执行 trim → 运行 graph → 写回 Redis。

---

## FastAPI 接口

### POST /api/trip/plan（改造现有）

```
body:  { user_id: str } + TripRequest 原有字段
response: TripPlanResponse
```

生成行程后将初始 state 写入 Redis。

### POST /api/chat/modify（新增）

```
body:  { user_id: str, message: str }
response: { reply: str, updated_plan: TripPlan }
```

从 Redis 读取 state，无需前端回传 `current_plan`。

---

## 前端改动（Vue）

**新增组件：** `TripModifyChat.vue`

```
┌─────────────────────────────────┬──────────────────┐
│  行程展示（现有 TripPlan 渲染）   │  💬 修改行程      │
│                                 │ ──────────────── │
│  第一天：故宫 → 天坛 → ...       │ 用户: 把第二天    │
│  第二天：...                    │       改成爬长城  │
│                                 │ AI: 已更新第二天  │
│                                 │     行程...      │
│                                 │ ──────────────── │
│                                 │ [输入框] [发送]   │
└─────────────────────────────────┴──────────────────┘
```

- `localStorage` 存储 `user_id`（UUID），首次访问自动生成
- 收到 `updated_plan` 后 `emit('plan-updated', updatedPlan)` 更新父组件行程数据

---

## 简历对应关系

| 简历条目 | 实现方式 |
|---------|---------|
| Multi-Agent Supervisor + SubGraph 状态隔离 | 独立 `TypedDict` State + `StateGraph.compile()` |
| 滑动窗口 + 动态摘要 State 裁剪 | `trim_state()` + `state.summary` 字段 |
| 跨供应商降级链路（DeepSeek/Gemini/OpenAI） | PROVIDERS 列表 + 调用点 try/except |
| 并行分发 18s→6s | LangGraph `Send()` API 原生并行 |

---

## 依赖变更

**新增：**
```
langchain>=0.3
langchain-openai>=0.2
langgraph>=0.2
langchain-mcp-adapters>=0.1
redis[asyncio]>=5.0
```

**移除：**
```
hello-agents[protocols]
fastmcp
huggingface_hub
```
