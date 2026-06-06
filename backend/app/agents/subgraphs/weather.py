import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import WeatherSubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import acall_with_fallback
from app.models.schemas import WeatherInfo


async def fetch_weather(state: WeatherSubState) -> dict:
    tool = next((t for t in get_amap_tools() if "weather" in t.name.lower()), None)
    if tool is None:
        return {"raw_result": "[]"}
    result = await tool.ainvoke({"city": state["city"]})
    return {"raw_result": str(result)}


async def parse_weather(state: WeatherSubState) -> dict:
    prompt = [
        SystemMessage(content=(
            "从高德天气查询结果中提取天气数据，返回 JSON 数组。\n"
            '每项格式：{"date":"YYYY-MM-DD","day_weather":"晴","night_weather":"多云",'
            '"day_temp":28,"night_temp":18,"wind_direction":"南","wind_power":"3级"}\n'
            "只返回 JSON 数组，不要其他文字。"
        )),
        HumanMessage(content=(
            f"查询结果：{state['raw_result']}\n"
            f"目标日期：{state['travel_dates']}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    try:
        data = json.loads(response.content)
        return {"weather_result": [WeatherInfo(**item) for item in data]}
    except Exception:
        return {"weather_result": []}


def create_weather_subgraph():
    g = StateGraph(WeatherSubState)
    g.add_node("fetch", fetch_weather)
    g.add_node("parse", parse_weather)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


weather_subgraph = create_weather_subgraph()
