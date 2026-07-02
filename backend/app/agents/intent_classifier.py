import re
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.llm_router import get_agent_llm

Intent = Literal["query_plan", "modify", "other"]

QUERY_RULES: list[str] = [
    r"(住哪|酒店|住宿)",
    r"(景点|去哪|参观|游览)",
    r"(天气|温度|气温)",
    r"(预算|费用|花多少|多少钱)",
    r"(餐|吃什么|午餐|晚餐|早餐)",
]

OTHER_RULES: list[str] = [
    r"^(谢谢|感谢|好的|可以|没问题|好|嗯|收到)[！!。]*$",
    r"^(你好|您好|hi|hello)[！!。]*$",
]

_CLASSIFIER_PROMPT = (
    "你是行程助手的意图分类器。根据用户消息，判断意图：\n"
    "- query_plan：查询当前行程信息（天气、景点、酒店、餐饮、预算等）\n"
    "- modify：修改、调整、新增或删除行程内容\n"
    "- other：闲聊、问候、感谢等与行程无关的内容\n"
    "返回结构化 JSON，包含 intent 和 confidence（0.0-1.0）。"
)


class IntentResult(BaseModel):
    intent: Intent
    confidence: float


def classify_by_rules(message: str) -> Intent | None:
    for pattern in QUERY_RULES:
        if re.search(pattern, message):
            return "query_plan"
    for pattern in OTHER_RULES:
        if re.search(pattern, message, re.IGNORECASE):
            return "other"
    return None


async def classify_intent(message: str) -> Intent:
    rule_result = classify_by_rules(message)
    if rule_result is not None:
        return rule_result
    try:
        llm = get_agent_llm("intent_classifier")
        structured = llm.with_structured_output(IntentResult, method="function_calling")
        result: IntentResult = await structured.ainvoke([
            SystemMessage(content=_CLASSIFIER_PROMPT),
            HumanMessage(content=message),
        ])
        if result.confidence < 0.7:
            return "modify"
        return result.intent
    except Exception:
        return "modify"
