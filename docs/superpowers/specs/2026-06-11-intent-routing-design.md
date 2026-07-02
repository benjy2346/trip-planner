# Intent Routing Design

**Date:** 2026-06-11  
**Branch:** feature/langgraph-refactor  
**Goal:** 通过规则层 + LLM 分类减少 `modify_trip` 端点的无效 LLM 调用，同时把对话流程迁移进 LangGraph。

---

## Background

当前 `/api/chat/modify` 对所有用户消息无差别调用完整 LLM 修改流程，包括查询行程信息、闲聊等不需要修改 `trip_plan` 的场景。引入意图路由后，只有真正的修改请求才触发 LLM。

---

## Architecture

### 两个 graph，共享 checkpointer

```
/api/trip/plan   → supervisor_graph → 写入 trip_plan + messages
/api/chat/modify → chat_graph       → 读取 trip_plan，追加 messages
```

两个 graph 使用相同的 `thread_id = user_id` 和同一个 `AsyncRedisSaver` 实例，通过 checkpointer 共享 state，互不感知对方的内部逻辑。

### chat_graph 流程

```
START
  ↓
classify_intent
  ↓  (Command API，直接指定下一跳，不需 conditional edge)
  ├── "query_handler"
  ├── "modify_handler"
  └── "other_handler"
        ↓
       END
```

节点使用 LangGraph `Command` 返回路由指令：

```python
async def classify_intent_node(state: SupervisorState) -> Command:
    intent = await classify(state["messages"][-1].content, state)
    return Command(goto=intent)
```

---

## Intent Classification

### 三类意图

| 意图 | 描述 | 执行路径 |
|------|------|---------|
| `query_plan` | 查询当前行程信息 | 程序化从 state 提取，0 LLM |
| `modify` | 修改、调整、新增、删除行程 | 完整 LLM 修改流程 |
| `other` | 闲聊、问候、感谢等无关内容 | 固定文案，0 LLM |

### 规则层（Layer 1）

正则匹配，0 cost，命中即路由：

```python
QUERY_RULES = [
    r"第[一二三四五六七八九十\d]+天",
    r"(住哪|酒店|住宿)",
    r"(景点|去哪|参观|游览)",
    r"(天气|温度|气温)",
    r"(预算|费用|花多少|多少钱)",
    r"(餐|吃什么|午餐|晚餐|早餐)",
]

OTHER_RULES = [
    r"^(谢谢|感谢|好的|可以|没问题|好|嗯|收到)[\！!。]*$",
    r"^(你好|您好|hi|hello)[\！!。]*$",
]
```

未命中则进入 LLM 分类层。

### LLM 分类层（Layer 2）

使用 `agents_config.yaml` 中 `intent_classifier` 配置的模型（`deepseek-chat`，`temperature=0`），返回结构化输出：

```python
class IntentResult(BaseModel):
    intent: Literal["query_plan", "modify", "other"]
    confidence: float
```

`confidence < 0.7` 时兜底路由到 `modify`，宁可多调一次也不误判。

---

## Nodes

### `query_handler`

程序化从 `trip_plan` 取值，不调 LLM。根据用户消息提取参数：

| 匹配模式 | 返回数据 |
|---------|---------|
| 第N天 + 酒店 | `days[N-1].hotel` |
| 第N天 + 景点 | `days[N-1].attractions` |
| 第N天 + 餐 | `days[N-1].meals` |
| 天气 | `weather_info` |
| 预算/费用 | `budget` |
| 无法提取 | 兜底文案，引导用户明确问题 |

输出：`{"messages": [AIMessage(reply)]}`

### `modify_handler`

将现有 `chat.py` 中 `modify_trip` 的 LLM 调用逻辑原样迁移，替换为 `get_agent_llm("modify_handler")`。`trim_state` 逻辑保留在此节点内。

输出：`{"messages": [AIMessage(reply)], "trip_plan": updated_plan}`（HumanMessage 在 `ainvoke` 时传入，节点只追加 AI 回复）

### `other_handler`

固定文案，0 LLM 调用：

```
"我是行程助手，只能帮您查询或修改当前行程。请告诉我您想了解或修改什么？"
```

输出：`{"messages": [AIMessage(canned_response)]}`

---

## `agents_config.yaml`

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

Provider 的 `base_url` 和 `api_key` 从 `get_settings()` 动态注入，不写在 yaml 里。

---

## `llm_router.py` 扩展

新增 `get_agent_llm(agent_name: str) -> ChatOpenAI`：

```python
def get_agent_llm(agent_name: str) -> ChatOpenAI:
    config = _load_agents_config()["agents"][agent_name]
    settings = get_settings()
    base_url = getattr(settings, f"{config['provider']}_base_url")
    api_key  = getattr(settings, f"{config['provider']}_api_key")
    return _make_llm(base_url, api_key, config["model"], config.get("temperature", 0.7))
```

结果按 `agent_name` 缓存，避免重复创建。

---

## Files

| 操作 | 文件 |
|------|------|
| 新建 | `backend/agents_config.yaml` |
| 新建 | `backend/app/agents/chat_graph.py` |
| 新建 | `backend/app/agents/intent_classifier.py` |
| 修改 | `backend/app/agents/llm_router.py` |
| 修改 | `backend/app/agents/__init__.py` |
| 修改 | `backend/app/api/main.py` |
| 修改 | `backend/app/api/routes/chat.py` |

---

## Error Handling

- 规则层无异常路径（纯 Python）
- LLM 分类失败（网络/超时）→ 兜底路由到 `modify`，不抛错
- `query_handler` 提取失败 → 返回引导文案，不抛错
- `modify_handler` LLM 失败 → 走现有 `acall_with_fallback` 降级链
