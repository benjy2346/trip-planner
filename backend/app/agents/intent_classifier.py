import re
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.llm_router import get_agent_llm
from app.agents import intent_model
from app.agents.intent_model import IntentModelUnavailable

Intent = Literal["query_weather", "query_attraction", "query_hotel", "plan_change", "other"]

CONFIDENCE_THRESHOLD = 0.7

QUERY_WEATHER_RULES: list[str] = [r"(天气|温度|气温|下雨|冷不冷|热不热)"]
QUERY_HOTEL_RULES: list[str] = [
    r"(住哪|酒店|住宿|宾馆)",
    r"(预算|费用|花多少|多少钱|总价)",
]
QUERY_ATTRACTION_RULES: list[str] = [
    r"(景点|去哪|参观|游览|好玩)",
    r"(餐|吃什么|午餐|晚餐|早餐|餐厅)",
]
OTHER_RULES: list[str] = [
    r"^(谢谢|感谢|好的|可以|没问题|好|嗯|收到)[！!。]*$",
    r"^(你好|您好|hi|hello)[！!。]*$",
]

_CLASSIFIER_PROMPT = (
    "你是行程助手的意图分类器。根据用户消息，判断意图：\n"
    "- query_weather：查询天气、气温、是否下雨、穿衣建议\n"
    "- query_attraction：查询景点、游玩安排、游览时长、餐饮/吃饭\n"
    "- query_hotel：查询住宿/酒店、酒店位置评分、住宿费用与总预算\n"
    "- plan_change：生成新行程，或修改、增删、调整已有行程\n"
    "- other：问候、感谢、闲聊或与行程无关的内容\n"
    "返回结构化 JSON，包含 intent 和 confidence（0.0-1.0）。"
)


class IntentResult(BaseModel):
    intent: Intent
    confidence: float


def classify_by_rules(message: str) -> Intent | None:
    for pattern in QUERY_WEATHER_RULES:
        if re.search(pattern, message):
            return "query_weather"
    for pattern in QUERY_HOTEL_RULES:
        if re.search(pattern, message):
            return "query_hotel"
    for pattern in QUERY_ATTRACTION_RULES:
        if re.search(pattern, message):
            return "query_attraction"
    for pattern in OTHER_RULES:
        if re.search(pattern, message, re.IGNORECASE):
            return "other"
    return None


async def _classify_by_llm(message: str) -> Intent:
    try:
        llm = get_agent_llm("intent_classifier")
        structured = llm.with_structured_output(IntentResult, method="function_calling")
        result: IntentResult = await structured.ainvoke([
            SystemMessage(content=_CLASSIFIER_PROMPT),
            HumanMessage(content=message),
        ])
        if result.confidence < CONFIDENCE_THRESHOLD:
            return "plan_change"
        return result.intent
    except Exception:
        return "plan_change"


async def classify_intent(message: str) -> Intent:
    rule_result = classify_by_rules(message)
    if rule_result is not None:
        return rule_result

    try:
        label, confidence = intent_model.predict(message)
        if confidence >= CONFIDENCE_THRESHOLD:
            return label  # type: ignore[return-value]
    except IntentModelUnavailable:
        pass
    except Exception:
        pass

    return await _classify_by_llm(message)
