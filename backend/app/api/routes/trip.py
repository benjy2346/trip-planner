"""旅行规划API路由"""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import RemoveMessage
from app.models.schemas import TripRequest, TripPlanResponse
from app.agents import get_supervisor_graph
from app.agents.state import SupervisorState

from ...logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/trip", tags=["旅行规划"])


@router.post("/plan", response_model=TripPlanResponse, summary="生成旅行计划")
async def plan_trip(request: TripRequest):
    try:
        logger.info("收到旅行规划请求: %s %s天 [用户:%s]", request.city, request.travel_days, request.user_id[:8])

        graph = get_supervisor_graph()
        config = {"configurable": {"thread_id": request.user_id}}

        # 重新规划时清空历史 messages，避免跨 session 累积
        snapshot = await graph.aget_state(config)
        if snapshot and snapshot.values.get("messages"):
            await graph.aupdate_state(
                config,
                {"messages": [RemoveMessage(id=m.id) for m in snapshot.values["messages"]]},
            )

        initial_state = SupervisorState(
            trip_request=request,
            messages=[],
            trip_plan=None,
            summary="",
            weather_outputs=[],
            hotel_outputs=[],
            poi_outputs=[],
        )
        result = await graph.ainvoke(initial_state, config=config)

        logger.info("旅行计划生成成功")
        return TripPlanResponse(success=True, message="旅行计划生成成功", data=result["trip_plan"])

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {e}")


@router.get("/health", summary="健康检查")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-langgraph"}
