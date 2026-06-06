import json
from datetime import date, timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.agents.state import SupervisorState, WeatherSubState, HotelSubState, POISubState
from app.agents.subgraphs.weather import weather_subgraph
from app.agents.subgraphs.hotel import hotel_subgraph
from app.agents.subgraphs.poi import poi_subgraph
from app.agents.llm_router import acall_with_fallback
from app.models.schemas import TripPlan


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
    prompt = [
        SystemMessage(content=(
            "你是旅行规划助手。根据提供的天气、酒店、景点数据生成详细行程，返回 JSON。\n"
            '格式：{"city":"...","start_date":"...","end_date":"...","days":[...],"overall_suggestions":"..."}\n'
            "只返回 JSON，不要其他文字。"
        )),
        HumanMessage(content=(
            f"城市：{req.city}，{req.start_date}~{req.end_date}，{req.travel_days}天\n"
            f"交通：{req.transportation}，住宿：{req.accommodation}\n"
            f"偏好：{req.preferences}\n"
            f"{f'额外要求：{req.free_text_input}' if req.free_text_input else ''}\n\n"
            f"天气：{[w.model_dump() for w in state.get('weather_outputs', [])]}\n"
            f"酒店：{[h.model_dump() for h in state.get('hotel_outputs', [])]}\n"
            f"景点：{[p.model_dump() for p in state.get('poi_outputs', [])]}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    try:
        data = json.loads(response.content)
        trip_plan = TripPlan(**data)
    except Exception:
        trip_plan = TripPlan(
            city=req.city,
            start_date=req.start_date,
            end_date=req.end_date,
            days=[],
            overall_suggestions="行程生成失败，请重试",
        )
    return {
        "trip_plan": trip_plan,
        "messages": [AIMessage(content=f"已为您生成{req.city}{req.travel_days}天行程。")],
    }


def create_supervisor_graph():
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
    return builder.compile()


supervisor_graph = create_supervisor_graph()
