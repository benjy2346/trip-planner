import asyncio
import json
from datetime import date, timedelta

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from app.agents.state import SupervisorState
from app.agents.llm_router import acall_agent_with_fallback
from app.config import get_settings
from app.models.schemas import TripPlan
from app.planner.context import PlannerContextBuilder, build_grounded_planner_messages
from app.planner.validation import validate_grounded_trip_plan, recompute_grounded_budget


_planner_builder = PlannerContextBuilder(amap_api_key=get_settings().amap_api_key)


def _date_range(start: str, days: int) -> list[str]:
    """行程日期序列。仅存量 ml 脚本（build_eval_set）还在引用，保留兼容。"""
    d = date.fromisoformat(start)
    return [(d + timedelta(days=i)).isoformat() for i in range(days)]


async def assembler_node(state: SupervisorState) -> dict:
    req = state["trip_request"]
    # collect() 内部用线程池并行取高德结构化数据；它是同步的，放到线程里跑，
    # 不阻塞事件循环。取数并发已下沉到 Builder，不再需要图层面的子图扇出。
    context = await asyncio.to_thread(_planner_builder.collect, req)
    compact = _planner_builder.compact_for_planner(context)

    response = await acall_agent_with_fallback("assembler", build_grounded_planner_messages(compact))
    content = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    trip_plan = TripPlan(**json.loads(content))

    violations = validate_grounded_trip_plan(trip_plan, context)
    if violations:
        print(f"⚠️ TripPlan 校验告警 {len(violations)} 条: {violations[:5]}")
    trip_plan.budget = recompute_grounded_budget(trip_plan, context["party"]["total"])

    return {
        "trip_plan": trip_plan,
        "messages": [AIMessage(content=f"已为您生成{req.city}{req.travel_days}天行程。")],
    }


def create_supervisor_graph(checkpointer=None):
    builder = StateGraph(SupervisorState)
    builder.add_node("assembler", assembler_node)
    builder.add_edge(START, "assembler")
    builder.add_edge("assembler", END)
    return builder.compile(checkpointer=checkpointer)
