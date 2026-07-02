import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import POISubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import acall_with_fallback
from app.models.schemas import Attraction, Location


async def fetch_pois(state: POISubState) -> dict:
    tool = next((t for t in get_amap_tools() if "search" in t.name.lower()), None)
    if tool is None:
        return {"raw_result": "[]"}
    keywords = "、".join(state["preferences"]) if state["preferences"] else "热门景点"
    result = await tool.ainvoke({"keywords": keywords, "city": state["city"]})
    return {"raw_result": str(result)}


async def parse_pois(state: POISubState) -> dict:
    prompt = [
        SystemMessage(content=(
            "从高德 POI 搜索结果中提取景点信息，返回 JSON 数组。\n"
            '每项格式：{"name":"...","address":"...","location":{"longitude":0.0,"latitude":0.0},'
            '"visit_duration":120,"description":"...","category":"...","rating":4.5,"ticket_price":0}\n'
            "只返回 JSON 数组，不要其他文字。"
        )),
        HumanMessage(content=(
            f"搜索结果：{state['raw_result']}\n"
            f"偏好：{state['preferences']}，天数：{state['travel_days']}，城市：{state['city']}"
        )),
    ]
    response = await acall_with_fallback(prompt)
    data = json.loads(response.content)
    attractions = []
    for item in data:
        clean = {k: v for k, v in item.items() if v is not None}
        if "location" not in clean:
            clean["location"] = {"longitude": 0.0, "latitude": 0.0}
        attractions.append(Attraction(**clean))
    return {"poi_result": attractions}


def create_poi_subgraph():
    g = StateGraph(POISubState)
    g.add_node("fetch", fetch_pois)
    g.add_node("parse", parse_pois)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


poi_subgraph = create_poi_subgraph()
