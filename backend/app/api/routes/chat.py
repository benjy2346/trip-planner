"""多轮对话修改行程 API"""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.agents import get_supervisor_graph
from app.agents.llm_router import acall_with_fallback, get_primary_llm
from app.agents.state_trimmer import trim_state
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
    graph = get_supervisor_graph()
    config = {"configurable": {"thread_id": request.user_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在，请先生成行程")

    state = snapshot.values
    # trim 仅用于构造 LLM prompt，不替换 Redis 中的完整历史
    trimmed = trim_state(state, get_primary_llm())

    # 若摘要发生更新，持久化 summary 字段
    if trimmed.get("summary") != state.get("summary"):
        await graph.aupdate_state(config, {"summary": trimmed["summary"]}, as_node="assembler")

    summary_ctx = f"历史摘要：{trimmed['summary']}\n" if trimmed.get("summary") else ""
    current_plan = state["trip_plan"].model_dump_json() if state.get("trip_plan") else "无"

    prompt = [
        SystemMessage(content=(
            f"你是旅行修改助手。{summary_ctx}"
            f"当前行程 JSON：{current_plan}\n"
            '根据用户请求修改行程，返回 JSON：{"reply":"...","updated_plan":{...}}。'
            "如无需修改行程结构只需口头回答，updated_plan 返回原值。只返回 JSON。"
        )),
        *trimmed["messages"],
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

    # add_messages reducer 会自动追加，只传新增的两条消息即可
    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage(content=request.message), AIMessage(content=reply)],
            "trip_plan": updated_plan,
        },
        as_node="assembler",
    )

    return ChatModifyResponse(reply=reply, updated_plan=updated_plan)
