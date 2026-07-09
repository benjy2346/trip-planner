from datetime import date, timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.agents.state import SupervisorState, WeatherSubState, HotelSubState, POISubState
from app.agents.subgraphs.weather import weather_subgraph
from app.agents.subgraphs.hotel import hotel_subgraph
from app.agents.subgraphs.poi import poi_subgraph
import json
from app.agents.llm_router import acall_with_fallback, acall_agent_with_fallback
from app.models.schemas import TripPlan
from app.planner.context import build_planner_context, build_planner_messages
from app.planner.validation import validate_trip_plan, recompute_budget


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
    context = build_planner_context(
        req,
        state.get("weather_outputs", []),
        state.get("hotel_outputs", []),
        state.get("poi_outputs", []),
    )
    response = await acall_agent_with_fallback("assembler", build_planner_messages(context))
    content = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    trip_plan = TripPlan(**json.loads(content))

    violations = validate_trip_plan(trip_plan, context)
    if violations:
        print(f"⚠️ TripPlan 校验告警 {len(violations)} 条: {violations[:5]}")
    trip_plan.budget = recompute_budget(trip_plan, context["party"]["total"])

    return {
        "trip_plan": trip_plan,
        "messages": [AIMessage(content=f"已为您生成{req.city}{req.travel_days}天行程。")],
    }


def create_supervisor_graph(checkpointer=None):
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
    return builder.compile(checkpointer=checkpointer)
