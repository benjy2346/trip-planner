from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import WeatherSubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import get_structured_chain
from app.models.schemas import WeatherInfo


class _WeatherOutput(BaseModel):
    weather_result: list[WeatherInfo]


async def fetch_weather(state: WeatherSubState) -> dict:
    tool = next((t for t in get_amap_tools() if "weather" in t.name.lower()), None)
    if tool is None:
        return {"raw_result": "[]"}
    result = await tool.ainvoke({"city": state["city"]})
    return {"raw_result": str(result)}


async def parse_weather(state: WeatherSubState) -> dict:
    prompt = [
        SystemMessage(content=(
            "从高德天气查询结果中提取天气数据，"
            "按 weather_result 字段返回列表，每项包含 date、day_weather、"
            "night_weather、day_temp、night_temp、wind_direction、wind_power。"
        )),
        HumanMessage(content=(
            f"查询结果：{state['raw_result']}\n"
            f"目标日期：{state['travel_dates']}"
        )),
    ]
    result = await get_structured_chain(_WeatherOutput).ainvoke(prompt)
    return {"weather_result": result.weather_result}


def create_weather_subgraph():
    g = StateGraph(WeatherSubState)
    g.add_node("fetch", fetch_weather)
    g.add_node("parse", parse_weather)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


weather_subgraph = create_weather_subgraph()
