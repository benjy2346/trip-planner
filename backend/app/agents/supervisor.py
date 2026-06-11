from datetime import date, timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.agents.state import SupervisorState, WeatherSubState, HotelSubState, POISubState
from app.agents.subgraphs.weather import weather_subgraph
from app.agents.subgraphs.hotel import hotel_subgraph
from app.agents.subgraphs.poi import poi_subgraph
import json
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
    PLANNER_SYSTEM_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**
8. 只返回JSON，不要其他文字，不要markdown代码块"""

    prompt = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"城市：{req.city}，{req.start_date}~{req.end_date}，{req.travel_days}天\n"
            f"交通：{req.transportation}，住宿：{req.accommodation}\n"
            f"偏好：{req.preferences}\n"
            f"{f'额外要求：{req.free_text_input}' if req.free_text_input else ''}\n\n"
            f"天气数据：{[w.model_dump() for w in state.get('weather_outputs', [])]}\n"
            f"推荐酒店：{[h.model_dump() for h in state.get('hotel_outputs', [])]}\n"
            f"推荐景点：{[p.model_dump() for p in state.get('poi_outputs', [])]}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    content = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(content)
    trip_plan = TripPlan(**data)
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
