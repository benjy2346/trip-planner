"""多轮对话修改行程 API"""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.agents.llm_router import acall_with_fallback, get_primary_llm
from app.agents.state_trimmer import trim_state
from app.services.session_store import load_session, save_session
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
    state = await load_session(user_id)
    if state is None:
        return {"messages": []}
    msgs = [
        {"role": "user" if m.type == "human" else "ai", "content": m.content}
        for m in state.get("messages", [])
    ]
    return {"messages": msgs}


@router.post("/modify", response_model=ChatModifyResponse, summary="多轮修改行程")
async def modify_trip(request: ChatModifyRequest):
    state = await load_session(request.user_id)
    if state is None:
        raise HTTPException(status_code=404, detail="会话不存在，请先生成行程")

    state = trim_state(state, get_primary_llm())

    summary_ctx = f"历史摘要：{state['summary']}\n" if state.get("summary") else ""
    current_plan = state["trip_plan"].model_dump_json() if state.get("trip_plan") else "无"

    prompt = [
        SystemMessage(content=(
            f"你是旅行修改助手。{summary_ctx}"
            f"当前行程 JSON：{current_plan}\n"
            '根据用户请求修改行程，返回 JSON：{"reply":"...","updated_plan":{...}}。'
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
