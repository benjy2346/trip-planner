from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import POISubState
from app.services.amap_tools import get_amap_tools
from app.agents.llm_router import get_structured_chain
from app.models.schemas import Attraction


class _POIOutput(BaseModel):
    poi_result: list[Attraction]


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
            "从高德 POI 搜索结果中提取景点信息，"
            "按 poi_result 字段返回列表，每项包含 name、address、location（longitude/latitude）、"
            "visit_duration、description、category、rating、ticket_price。"
        )),
        HumanMessage(content=(
            f"搜索结果：{state['raw_result']}\n"
            f"偏好：{state['preferences']}，天数：{state['travel_days']}，城市：{state['city']}"
        )),
    ]
    result = await get_structured_chain(_POIOutput).ainvoke(prompt)
    return {"poi_result": result.poi_result}


def create_poi_subgraph():
    g = StateGraph(POISubState)
    g.add_node("fetch", fetch_pois)
    g.add_node("parse", parse_pois)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", END)
    return g.compile()


poi_subgraph = create_poi_subgraph()
