"""旅行规划API路由"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import TripRequest, TripPlanResponse
from app.agents import supervisor_graph
from app.agents.state import SupervisorState
from app.services.session_store import save_session

router = APIRouter(prefix="/trip", tags=["旅行规划"])


@router.post("/plan", response_model=TripPlanResponse, summary="生成旅行计划")
async def plan_trip(request: TripRequest):
    try:
        print(f"\n{'='*60}")
        print(f"📥 收到旅行规划请求: {request.city} {request.travel_days}天 [用户:{request.user_id[:8]}]")
        print(f"{'='*60}\n")

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

        print("✅ 旅行计划生成成功\n")
        return TripPlanResponse(success=True, message="旅行计划生成成功", data=result["trip_plan"])

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {e}")


@router.get("/health", summary="健康检查")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-langgraph"}
