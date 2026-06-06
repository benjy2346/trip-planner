import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import HotelSubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import acall_with_fallback
from app.models.schemas import Hotel


async def fetch_hotels(state: HotelSubState) -> dict:
    tool = next((t for t in get_amap_tools() if "search" in t.name.lower()), None)
    if tool is None:
        return {"raw_result": "[]"}
    result = await tool.ainvoke({
        "keywords": f"{state['accommodation_pref']}酒店",
        "city": state["city"],
    })
    return {"raw_result": str(result)}


async def parse_hotels(state: HotelSubState) -> dict:
    prompt = [
        SystemMessage(content=(
            "从高德 POI 搜索结果中提取酒店信息，返回 JSON 数组（最多 3 家）。\n"
            '每项格式：{"name":"...","address":"...","price_range":"...","rating":"...","distance":"...","type":"...","estimated_cost":0}\n'
            "只返回 JSON 数组。"
        )),
        HumanMessage(content=(
            f"搜索结果：{state['raw_result']}\n"
            f"偏好：{state['accommodation_pref']}，城市：{state['city']}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    try:
        data = json.loads(response.content)
        return {"hotel_result": [Hotel(**item) for item in data]}
    except Exception:
        return {"hotel_result": []}


def create_hotel_subgraph():
    g = StateGraph(HotelSubState)
    g.add_node("fetch", fetch_hotels)
    g.add_node("parse", parse_hotels)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


hotel_subgraph = create_hotel_subgraph()
