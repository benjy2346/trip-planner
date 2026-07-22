"""统一的行程图：生成与对话共用一张图、一份状态、一个 checkpointer。

此前生成走 supervisor_graph、对话走 chat_graph，两张图靠"恰好用了同一个
SupervisorState 和同一个 thread_id"隐式共享状态——没有任何代码声明这层关系，
`/chat/modify` 甚至要先从 supervisor_graph 掏 state 再交给 chat_graph 执行。

合并后入口由 `mode` 显式分流，共享关系变成结构本身。节点函数保持不变。
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from app.agents.chat_graph import (
    classify_intent_node,
    modify_handler_node,
    other_handler_node,
    query_handler_node,
)
from app.agents.state import SupervisorState
from app.agents.supervisor import assembler_node

MODE_GENERATE = "generate"
MODE_CHAT = "chat"


async def entry_router_node(state: SupervisorState) -> Command:
    """按调用方声明的 mode 分流；缺省按对话处理。"""
    mode = state.get("mode", MODE_CHAT)
    goto = "assembler" if mode == MODE_GENERATE else "classify_intent"
    return Command(goto=goto)


def create_trip_graph(checkpointer=None):
    builder = StateGraph(SupervisorState)

    builder.add_node("entry_router", entry_router_node)
    builder.add_node("assembler", assembler_node)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("query_handler", query_handler_node)
    builder.add_node("modify_handler", modify_handler_node)
    builder.add_node("other_handler", other_handler_node)

    builder.add_edge(START, "entry_router")
    # entry_router 与 classify_intent 都返回 Command(goto=...)，路由在节点内决定，
    # 因此这里只需声明各分支的出边。
    builder.add_edge("assembler", END)
    builder.add_edge("query_handler", END)
    builder.add_edge("modify_handler", END)
    builder.add_edge("other_handler", END)

    return builder.compile(checkpointer=checkpointer)
