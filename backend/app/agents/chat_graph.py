import json
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from app.agents.state import SupervisorState
from app.agents.intent_classifier import classify_intent
from app.agents.intent_labels import INTENT_TO_NODE, QUERY_INTENT_FIELD
from app.agents.llm_router import get_agent_llm
from app.models.schemas import TripPlan

_DAY_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _parse_day(message: str) -> int | None:
    m = re.search(r"第(\d+)天", message)
    if m:
        return int(m.group(1))
    m = re.search(r"第([一二三四五六七八九十])天", message)
    if m:
        return _DAY_MAP.get(m.group(1))
    return None


def _weather_summary(plan: TripPlan) -> str:
    if not plan.weather_info:
        return "暂无天气信息。"
    lines = [
        f"{w.date}：{w.day_weather}，白天 {w.day_temp}°C / 夜间 {w.night_temp}°C"
        for w in plan.weather_info
    ]
    return "天气预报：\n" + "\n".join(lines)


def _hotel_summary(plan: TripPlan) -> str:
    lines = [
        f"第{i}天：{d.hotel.name}（{d.hotel.address}），约 {d.hotel.estimated_cost} 元/晚"
        for i, d in enumerate(plan.days, 1) if d.hotel
    ]
    return "住宿安排：\n" + "\n".join(lines) if lines else "暂无酒店信息。"


def _attraction_summary(plan: TripPlan) -> str:
    lines = [
        f"第{i}天：{a.name}（建议 {a.visit_duration} 分钟）"
        for i, d in enumerate(plan.days, 1) for a in d.attractions
    ]
    return "景点安排：\n" + "\n".join(lines) if lines else "暂无景点信息。"


def _build_query_reply(message: str, state: SupervisorState, intent: str | None = None) -> str:
    plan: TripPlan | None = state.get("trip_plan")
    if not plan:
        return "还没有生成行程，请先规划行程。"

    if re.search(r"(天气|温度|气温)", message):
        return _weather_summary(plan)

    if re.search(r"(预算|费用|花多少|多少钱)", message):
        b = plan.budget
        if not b:
            return "暂无预算信息。"
        return (
            f"总预算：{b.total} 元\n"
            f"  景点门票：{b.total_attractions} 元\n"
            f"  住宿：{b.total_hotels} 元\n"
            f"  餐饮：{b.total_meals} 元\n"
            f"  交通：{b.total_transportation} 元"
        )

    day = _parse_day(message)
    if day is not None:
        idx = day - 1
        if idx < 0 or idx >= len(plan.days):
            return f"行程只有 {len(plan.days)} 天，没有第 {day} 天。"
        d = plan.days[idx]

        if re.search(r"(住哪|酒店|住宿)", message):
            h = d.hotel
            if not h:
                return "暂无酒店信息。"
            return f"第{day}天住宿：{h.name}（{h.address}），约 {h.estimated_cost} 元/晚。"

        if re.search(r"(餐|吃什么|午餐|晚餐|早餐)", message):
            if not d.meals:
                return "暂无餐饮信息。"
            lines = [f"  {m.type}：{m.name}，约 {m.estimated_cost} 元" for m in d.meals]
            return f"第{day}天餐饮：\n" + "\n".join(lines)

        if re.search(r"(景点|去哪|参观|游览)", message):
            if not d.attractions:
                return "暂无景点信息。"
            lines = [f"  {a.name}（建议 {a.visit_duration} 分钟）：{a.description[:30]}" for a in d.attractions]
            return f"第{day}天景点：\n" + "\n".join(lines)

        return f"第{day}天（{d.date}）：{d.description}"

    field = QUERY_INTENT_FIELD.get(intent or "")
    if field == "weather":
        return _weather_summary(plan)
    if field == "hotel":
        return _hotel_summary(plan)
    if field == "attraction":
        return _attraction_summary(plan)

    return "请问您想了解行程的哪部分？可以询问天气、预算、各天的景点、酒店或餐饮安排。"


async def query_handler_node(state: SupervisorState) -> dict:
    messages = state.get("messages", [])
    user_message = messages[-1].content if messages else ""
    reply = _build_query_reply(user_message, state, state.get("intent"))
    return {"messages": [AIMessage(content=reply)]}


async def other_handler_node(state: SupervisorState) -> dict:
    return {"messages": [AIMessage(
        content="我是行程助手，只能帮您查询或修改当前行程。请告诉我您想了解或修改什么？"
    )]}


async def modify_handler_node(state: SupervisorState) -> dict:
    from app.agents.state_trimmer import trim_state

    trimmed = trim_state(state, get_agent_llm("state_trimmer"))
    summary_ctx = f"历史摘要：{trimmed['summary']}\n" if trimmed.get("summary") else ""
    current_plan = state["trip_plan"].model_dump_json() if state.get("trip_plan") else "无"

    llm = get_agent_llm("modify_handler")
    prompt = [
        SystemMessage(content=(
            f"你是旅行修改助手。{summary_ctx}"
            f"当前行程 JSON：{current_plan}\n"
            '根据用户请求修改行程，返回 JSON：{"reply":"...","updated_plan":{...}}。'
            "如无需修改行程结构只需口头回答，updated_plan 返回原值。只返回 JSON。"
        )),
        *trimmed["messages"],
    ]

    response = await llm.ainvoke(prompt)

    try:
        data = json.loads(response.content)
        reply = data.get("reply", response.content)
        updated_plan_data = data.get("updated_plan")
        updated_plan = TripPlan(**updated_plan_data) if updated_plan_data else state.get("trip_plan")
    except Exception:
        reply = response.content
        updated_plan = state.get("trip_plan")

    update: dict = {
        "messages": [AIMessage(content=reply)],
        "trip_plan": updated_plan,
    }
    if trimmed.get("summary") != state.get("summary"):
        update["summary"] = trimmed["summary"]
    return update


async def classify_intent_node(state: SupervisorState) -> Command:
    messages = state.get("messages", [])
    if not messages:
        return Command(goto="other_handler")
    intent = await classify_intent(messages[-1].content)
    return Command(goto=INTENT_TO_NODE[intent], update={"intent": intent})


