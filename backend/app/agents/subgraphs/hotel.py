from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import HotelSubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import get_structured_chain
from app.models.schemas import Hotel


class _HotelOutput(BaseModel):
    hotel_result: list[Hotel]


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
            "从高德 POI 搜索结果中提取酒店信息（最多 3 家），"
            "按 hotel_result 字段返回列表，每项包含 name、address、"
            "price_range、rating、distance、type、estimated_cost。"
        )),
        HumanMessage(content=(
            f"搜索结果：{state['raw_result']}\n"
            f"偏好：{state['accommodation_pref']}，城市：{state['city']}"
        )),
    ]
    result = await get_structured_chain(_HotelOutput).ainvoke(prompt)
    return {"hotel_result": result.hotel_result}


def create_hotel_subgraph():
    g = StateGraph(HotelSubState)
    g.add_node("fetch", fetch_hotels)
    g.add_node("parse", parse_hotels)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


hotel_subgraph = create_hotel_subgraph()
