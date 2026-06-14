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
